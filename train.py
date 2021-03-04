from pi_gan_pytorch.pi_gan_pytorch import piGAN, Trainer

gan = piGAN(
    image_size = 128,
    dim = 512
).cuda()

trainer = Trainer(
    gan = gan,
    folder = 'images/img_align_celeba'
)

trainer()