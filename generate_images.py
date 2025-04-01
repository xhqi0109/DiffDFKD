import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig
import torchvision
from options.generate_images_adv_options import GenerateImagesAdvOptions
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

from helper.dataset import get_dataset
from helper.utils import kldiv, customer_aug_data, train_and_evaluate, load_classification_models, precompute_text_embeddings, generate_class_prompts

# Ensure required diffusers version is installed
check_min_version("0.10.0.dev0")
logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """Import appropriate text encoder class based on model architecture
    
    Args:
        pretrained_model_name_or_path: Name/path of pretrained model
        revision: Model revision version
    
    Returns:
        Text encoder class
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        torch_dtype=torch.float16,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def setup_accelerator_and_logging(args):
    """
    Initialize the accelerator and set up logging.
    """
    logging_dir = Path("./output_dir", "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="tensorboard",
        project_dir=logging_dir,
    )
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))
    return accelerator

def load_models_and_tokenizer(args, accelerator):
    """
    Load all necessary models and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
        torch_dtype=torch.float16
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        torch_dtype=torch.float16
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float16
    )
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="safety_checker",
        revision=args.revision,
        torch_dtype=torch.float16
    )
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="feature_extractor",
        revision=args.revision,
        torch_dtype=torch.float16
    )
    
    # Freeze models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Optimize CUDA settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Prepare for distributed training
    unet, text_encoder, tokenizer, generator, noise_scheduler, vae, safety_checker, feature_extractor = accelerator.prepare(
        unet, text_encoder, tokenizer, generator, noise_scheduler, vae, safety_checker, feature_extractor
    )
    
    # Set weight data type
    weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    return tokenizer, text_encoder, noise_scheduler, vae, unet, safety_checker, feature_extractor, generator, weight_dtype

def prepare_uncond_embeddings(tokenizer, text_encoder, unet, args):
    """
    Prepare unconditional embeddings.
    """
    uncond_inputs = tokenizer(
        ['' for _ in range(args.batch_size_generation)],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_input_ids = uncond_inputs.input_ids.to(unet.device)
    uncond_embeddings = text_encoder(uncond_input_ids)[0]
    return uncond_embeddings


def generate_images(accelerator, class_index, class_text_embeddings, uncond_embeddings, noise_scheduler, vae, unet, model, model_s,
                    hooks, transform, args, generator, weight_dtype, syn_image_seed):
    """
    Generate synthetic images for a single class with adversarial training.
    """
    image_save_dir_path = os.path.join(args.save_syn_data_path, str(class_index))
    os.makedirs(image_save_dir_path, exist_ok=True)
    
    text_embeddings = class_text_embeddings[class_index]
    
    # Set seed for reproducibility
    syn_image_seed += 1
    generator.manual_seed(syn_image_seed)
    set_seed(syn_image_seed)
    torch.cuda.empty_cache()

    with accelerator.accumulate(unet):
        latents_shape = (args.batch_size_generation, unet.in_channels, 64, 64)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device="cpu",
            dtype=weight_dtype
        ).to(unet.device)
        latents = latents * noise_scheduler.init_noise_sigma

        noise_scheduler.set_timesteps(args.inference_nums)
        timesteps_tensor = noise_scheduler.timesteps.to(latents.device)
        timestep_nums = 0

        for timesteps in timesteps_tensor[:-1]:
            # Enable gradients for necessary tensors
            text_embeddings.requires_grad_(True)
            uncond_embeddings.requires_grad_(True)
            latents.requires_grad_(True)

            input_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            latent_model_input = torch.cat([latents] * 2)

            model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
            uncond_pred, text_pred = model_preds.chunk(2)
            model_pred = uncond_pred + args.guided_scale * (text_pred - uncond_pred)

            # Calculate original latents
            with torch.no_grad():
                ori_latents = noise_scheduler.step(
                    model_pred,
                    timesteps.cpu(),
                    latents,
                    generator=generator
                ).pred_original_sample.half()

            input_latents = 1 / 0.18215 * ori_latents
            image = vae.decode(input_latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

            loss = 0.0
            loss_oh = loss_oh.new_zeros(1)
            loss_bn = loss_bn.new_zeros(1)
            loss_adv = loss_adv.new_zeros(1)

            if (args.oh + args.bn + args.adv) > 0:
                inputs_aug = transform(image)
                output = model(inputs_aug)

                # Calculate BatchNorm loss
                loss_bn = sum([h.r_feature for h in hooks])
                
                # Calculate one-hot loss
                target = torch.full((args.batch_size_generation,), class_index, device=unet.device, dtype=torch.long)
                loss_oh = F.cross_entropy(output, target)

                # Calculate adversarial loss
                if args.adv > 0 and args.class_per_num_start <= class_per_num:
                    s_out = model_s(inputs_aug)
                    mask = (s_out.argmax(1) == output.argmax(1)).float()
                    loss_adv = -(kldiv(s_out, output.detach(), reduction='none').sum(1) * mask).mean()

                loss = args.oh * loss_oh + args.bn * loss_bn + args.adv * loss_adv

                # Backpropagate loss
                cond_grad = torch.autograd.grad(loss, latents)[0]
                latents = latents - cond_grad

                # Save intermediate images
                if (timestep_nums + 1) % (args.inference_nums // 5) == 0 and (timestep_nums + 1) != args.inference_nums:
                    for i in range(args.batch_size_generation):
                        image_name = os.path.join(
                            image_save_dir_path,
                            f"{syn_image_seed}_{class_index}_s:{timesteps.item():.0f}_bn:{loss_bn.item():.3f}_oh:{loss_oh.item():.3f}_adv:{loss_adv.item():.3f}_{i}.jpg"
                        )
                        torchvision.utils.save_image(image[i], image_name)
            else:
                # Save intermediate images without loss
                if (timestep_nums + 1) % (args.inference_nums // 5) == 0 and (timestep_nums + 1) != args.inference_nums:
                    for i in range(args.batch_size_generation):
                        image_name = os.path.join(
                            image_save_dir_path,
                            f"{syn_image_seed}_{class_index}_s:{timesteps.item():.0f}_bn:0.0_oh:0.0_adv:0.0_{i}.jpg"
                        )
                        torchvision.utils.save_image(image[i], image_name)

            timestep_nums += 1

            with torch.no_grad():
                # Apply custom augmentation
                if args.customer_aug >= 1 and (timestep_nums % (args.inference_nums // 5) == 0) and (
                        timestep_nums != args.inference_nums):
                    latents, _, _ = customer_aug_data(latents, customer_aug=args.customer_aug)
                    latents = latents.to(dtype=weight_dtype)

                # Predict next latents
                latent_model_input = torch.cat([latents] * 2)
                model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                uncond_pred, text_pred = model_preds.chunk(2)
                model_pred = uncond_pred + args.guided_scale * (text_pred - uncond_pred)
                
                # Update latents
                latents = noise_scheduler.step(
                    model_pred,
                    timesteps.cpu(),
                    latents,
                    generator=generator
                ).prev_sample.half()

            # Clear gradients
            unet.zero_grad()
            vae.zero_grad()
            model.zero_grad()
            torch.cuda.empty_cache()

        # Save final images
        with torch.no_grad():
            ori_latents = noise_scheduler.step(
                model_pred,
                timesteps.cpu(),
                latents,
                generator=generator
            ).pred_original_sample.half()
            input_latents = 1 / 0.18215 * ori_latents.detach()
            image = vae.decode(input_latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

            for i in range(args.batch_size_generation):
                image_name = os.path.join(
                    image_save_dir_path,
                    f"{syn_image_seed}_{class_index}_bn:{loss_bn.item():.3f}_oh:{loss_oh.item():.3f}_adv:{loss_adv.item():.3f}_{i}.jpg"
                )
                torchvision.utils.save_image(image[i], image_name)
    
    return syn_image_seed

def main(args):
    """
    Main pipeline for adversarial image generation and model training.
    """
    accelerator = setup_accelerator_and_logging(args)
    tokenizer, text_encoder, noise_scheduler, vae, unet, safety_checker, feature_extractor, generator, weight_dtype = load_models_and_tokenizer(
        args, accelerator)
    uncond_embeddings = prepare_uncond_embeddings(tokenizer, text_encoder, unet, args)
    model, model_s, hooks, transform = load_classification_models(args, accelerator)
    class_prompts = generate_class_prompts(args)
    class_text_embeddings, class_syn_nums = precompute_text_embeddings(class_prompts, tokenizer, text_encoder, unet,
                                                                        args)

    # Create output directories
    args.save_syn_data_path = os.path.join(args.save_syn_data_path, args.name)
    os.makedirs(args.save_syn_data_path, exist_ok=True)
    args.train_data_path = args.save_syn_data_path
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    syn_image_seed = args.seed
    best_acc = -1.0
    for class_per_num in tqdm(range(int(args.generate_nums))):
        for class_index in range(len(class_prompts)):
            syn_image_seed = generate_images(accelerator, class_index, class_text_embeddings, uncond_embeddings, noise_scheduler, vae,
                                                unet, model, model_s, hooks, transform, args, generator, weight_dtype,
                                                syn_image_seed)

        # Train and evaluate every 10 generations
        if (class_per_num + 1) % 10 == 0:
            train_dataset, test_dataset = get_dataset(args)
            current_best_acc = train_and_evaluate(model, model_s, train_dataset, test_dataset, args, accelerator)
            if current_best_acc > best_acc:
                best_acc = current_best_acc
            if (class_per_num + 1) % 100 == 0:
                torch.save(
                    model_s.state_dict(),
                    os.path.join(args.checkpoints_dir, f'model-epoch:{class_per_num + 1}-{best_acc:.2f}.pt')
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()
    print(f"Final Best Accuracy: {best_acc:.2f}")

if __name__ == "__main__":
    args = GenerateImagesAdvOptions().parse()
    main(args)