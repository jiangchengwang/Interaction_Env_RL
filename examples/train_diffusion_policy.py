import json
import os
import sys
import pandas as pd

projector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(projector_dir)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.bev_dataset import BEVDataset
from diffusion_policy.networks import *
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import logger
log = logger.setup_app_level_logger(file_name="train_diffusion_policy.log")


def train_step(
        train_dataloader: torch.utils.data.DataLoader,
        nets: nn.ModuleDict,
        noise_scheduler: DDPMScheduler,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        ema: EMAModel,
) -> List[float]:
    epoch_loss = list()
    # batch loop
    with tqdm(train_dataloader, desc='Batch', leave=False) as tepoch:
        for nbatch in tepoch:
            # data normalized in dataset
            # device transfer
            nimage = nbatch['image'].to(device)
            naction = nbatch['action'].to(device)

            # encoder vision features
            image_features = nets['vision_encoder'](nimage)

            # concatenate vision feature and low-dim obs
            obs_cond = image_features

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (naction.shape[0],), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            tepoch.set_postfix(loss=loss_cpu)

        return epoch_loss


def eval_step(
        eval_dataloader: torch.utils.data.DataLoader,
        nets: nn.ModuleDict,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
):
    eval_loss = list()
    with torch.no_grad():
        with tqdm(eval_dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nimage = nbatch['image'].to(device)
                naction = nbatch['action'].to(device)

                # encoder vision features
                image_features = nets['vision_encoder'](nimage)

                # concatenate vision feature and low-dim obs
                obs_cond = image_features

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (naction.shape[0],), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)

                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                # logging
                loss_cpu = loss.item()
                eval_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
    return eval_loss


def save_exp_data(
    loss: List[float],
    save_path: str,
    iter: int,
    train=True,
):
    csv_file_name = "train_loss.csv" if train else "eval_loss.csv"
    csv_file_path = os.path.join(save_path, csv_file_name)
    exp_data = [{
        'mean': np.mean(loss),
        'std': np.std(loss),
        'max': np.max(loss),
        'min': np.min(loss),
        'epoch': iter,
    }]

    df = pd.DataFrame(exp_data)
    kwargs = {
        'index': False,
        'mode': 'w',
        'encoding': 'utf-8'
    }

    if iter > 0:
        kwargs['mode'] = 'a'
        kwargs['header'] = False

    df.to_csv(csv_file_path,**kwargs)


if __name__ == '__main__':

    arg = ArgumentParser()
    arg.add_argument('--dataset-path', type=str, default='../data/dataset', help="dataset path")
    arg.add_argument('--save-path', type=str, default='../data/model', help="dataset path")
    arg.add_argument('--epochs', type=int, default=50, help="train epochs")
    arg.add_argument('--batch-size', type=int, default=64, help="train batch size")
    arg.add_argument('--lr', type=float, default=1e-4, help="train learning rate")
    arg.add_argument('--weight-decay', type=float, default=1e-6, help="train weight decay")
    arg.add_argument('--num-workers', type=int, default=4, help="train num workers")
    arg.add_argument('--eval-interval', type=int, default=1, help="eval interval")
    args = arg.parse_args()
    log.info("Input args:")
    for k, v in vars(args).items():
        log.info(f"{k}: {v})")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, 'InputParm.json'), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_dataset = BEVDataset(dataset_path=args.dataset_path, train=True)
    eval_dataset = BEVDataset(dataset_path=args.dataset_path, train=False)

    # create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(train_dataloader))
    log.info(f"batch['image'].shape: {batch['image'].shape}")
    log.info(f"batch['action'].shape: {batch['action'].shape}")

    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    obs_dim = vision_feature_dim
    action_dim = (1, 5)

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim[0],
        global_cond_dim=obs_dim
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    num_epochs = args.epochs

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs
    )
    best_eval_loss = np.inf
    # epoch loop
    for epoch_idx in range(num_epochs):
        log.info(f"Epoch {epoch_idx}")
        train_loss = train_step(
            train_dataloader=train_dataloader,
            nets=nets,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            ema=ema,
        )
        log.info(f"Epoch {epoch_idx} loss: {np.mean(train_loss):.4f}")
        save_exp_data(
            loss=train_loss,
            save_path=args.save_path,
            iter=epoch_idx,
            train=True,
        )

        if epoch_idx % args.eval_interval == 0:
            eval_loss = eval_step(
                eval_dataloader=eval_dataloader,
                nets=nets,
                noise_scheduler=noise_scheduler,
                device=device,
            )
            log.info(f"Epoch {epoch_idx} eval loss: {np.mean(eval_loss):.4f}")
            save_exp_data(
                loss=eval_loss,
                save_path=args.save_path,
                iter=epoch_idx // args.eval_interval,
                train=False,
            )

            if np.mean(eval_loss) < best_eval_loss:
                best_eval_loss = np.mean(eval_loss)
                torch.save(nets.state_dict(), os.path.join(args.save_path, 'best_model.pt'))

        torch.save(nets.state_dict(), os.path.join(args.save_path, 'last_model.pt'))
        torch.save(ema.state_dict(), os.path.join(args.save_path, 'ema_model.pt'))