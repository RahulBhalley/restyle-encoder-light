import torch

def get_average_image(net, opts):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    if opts.dataset_type == "cars_encode":
        avg_image = avg_image[:, 32:224, :]
    return avg_image

def my_run_on_batch(x, net, opts, avg_image):
    y_hat, latent = None, None
    y_hats = {idx: {} for idx in range(x.shape[0])}
    
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
            x_input = torch.cat([x, avg_image_for_batch], dim=1)
            print(f"avg_image_for_batch: {avg_image_for_batch.shape}")
            print(f"x_input: {x_input.shape}")
            # latent will be None at start
            y_hat, latent = net.forward(x_input, latent=None, randomize_noise=False, return_latents=True, resize=opts.resize_outputs)
            print(f"y_hat: {y_hat.shape}")
            print(f"latent: {latent.shape}")
        else:
            x_input = torch.cat([x_input, y_hat], dim=1)
            y_hat, latent = net.forward(x_input, latent=latent, randomize_noise=False, return_latents=True, resize=opts.resize_outputs)
            
def run_on_batch(inputs, net, opts, avg_image):
    y_hat, latent = None, None
    results_batch = {idx: [] for idx in range(inputs.shape[0])}
    results_latent = {idx: [] for idx in range(inputs.shape[0])}
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
            print(f"[run_on_batch] avg_image_for_batch: {avg_image_for_batch.shape}")
            print(f"[run_on_batch] x_input: {x_input.shape}")
        else:
            print(f"[run_on_batch] inputs: {inputs.shape}")
            print(f"[run_on_batch] y_hat: {y_hat.shape}")
            x_input = torch.cat([inputs, y_hat], dim=1)
            print(f"[run_on_batch] x_input: {x_input.shape}")

        y_hat, latent = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)

        if opts.dataset_type == "cars_encode":
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        # store intermediate outputs
        for idx in range(inputs.shape[0]):
            results_batch[idx].append(y_hat[idx])
            results_latent[idx].append(latent[idx].cpu().numpy())

        # resize input to 256 before feeding into next iteration
        if opts.dataset_type == "cars_encode":
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    return results_batch, results_latent
