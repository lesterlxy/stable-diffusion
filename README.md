- txt2img
  - Uses slightly modified diffusion library from `https://github.com/crowsonkb/k-diffusion.git src/k-diffusion`
  - Added sigma schedulers
  - Added more command line options
  - Include face fixing model from `https://rentry.org/kretard` with memory shuffling so it doesn't eat up vram (running on 10GB)
  - Add weighted prompts from lstein/stable-diffusion
  - Add multiple init images and init image directory
  - Add variance correction in latent space when using multiple init images
  - Add umap
  - Add clip interrogator from `https://colab.research.google.com/github/pharmapsychotic/clip-interrogator`
- img2img
  - Has k-diffusers
  - Not a priority

# UMAP
Naive averaging in latent space:
![](assets/umap/0.png)
Average in umap components:
![](assets/umap/1.png)
Note preservation of common components without losing detail

# Todo
Tidy up accumulated mess in txt2imgkd