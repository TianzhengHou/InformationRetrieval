import os
import PIL
from transformers import CLIPProcessor, CLIPModel
from typing import List, Union
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import faiss

image_path = os.listdir('Images/')
image_path = ['Images/' + path for path in image_path if '.png' in path]
image_path.sort()
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_images(images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
    def transform_fn(el):
        if isinstance(el['image'], PIL.Image.Image):
            imgs = el['image']
        else:
            imgs = [Image().decode_example(_) for _ in el['image']]

        # returns preprocessed images using a predefined preprocess function from the CLIP model
        return preprocess(images=imgs, return_tensors='pt')

    # create a dataset object
    dataset = Dataset.from_dict({'image': images})
    dataset = dataset.cast_column('image',Image(decode=False)) if isinstance(images[0], str) else dataset
    # setting dataset object to pytorch format
    dataset.set_format('torch')
    # apply PIL transformation on images
    dataset.set_transform(transform_fn)

    # processing the dataset using a data loader for efficiently handling batches
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # processing batches and computing embeddings
    image_embeddings = []
    pbar = tqdm(total=len(images) // batch_size, position=0)
    # for each batch, we move the data to the GPU, and then we compute the image features
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v for k,v in batch.items()}
            # model gets the image features, then we detach the tensor from the GPU and move it to the CPU
            image_embeddings.extend(model.get_image_features(**batch).detach().cpu().numpy())
            pbar.update(1)
        pbar.close()
    return np.stack(image_embeddings)


# the function returns a numpy array with the image embeddings
vector_embedding = np.array(encode_images(image_path,128))

# similarly encode text
def encode_text( text: List[str], batch_size: int):
    # create a dataset object
    dataset = Dataset.from_dict({'text': text})
    dataset = dataset.map(lambda el: preprocess(text=el['text'], return_tensors="pt",
                                                        max_length=77, padding="max_length", truncation=True),
                            batched=True,
                            remove_columns=['text'])
    dataset.set_format('torch')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # generate text embeddings
    text_embeddings = []
    pbar = tqdm(total=len(text) // batch_size, position=0)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v for k, v in batch.items()}
            text_embeddings.extend(model.get_text_features(**batch).detach().cpu().numpy())
            pbar.update(1)
        pbar.close()
    return np.stack(text_embeddings)

with open('CLIP_image_embeddings.pkl','wb') as f:
    pickle.dump(vector_embedding, f)

import pickle
with open('CLIP_image_embeddings.pkl', 'rb') as fp:
    vector_embedding = pickle.load(fp)

# create an index
index = faiss.IndexFlatIP(vector_embedding.shape[1])
# add the image embeddings to the index
index.add(vector_embedding)

def Search(search_text, results):
  with torch.no_grad():
      text_search_embedding = encode_text([search_text], batch_size=32)
  text_search_embedding = text_search_embedding/np.linalg.norm(text_search_embedding, ord=2, axis=-1, keepdims=True)

  distances, indices = index.search(text_search_embedding.reshape(1, -1), results)

  distances = distances[0]
  indices = indices[0]
  indices_distances = list(zip(indices, distances))

  # Sort based on the distances
  indices_distances.sort(key=lambda x: x[1])


  from PIL import Image
  fixed_size = (300, 300)
  for idx, distance in indices_distances:
      path = image_path[idx]
      print(path)
      im = Image.open(path)
      im_resized = im.resize(fixed_size)
      plt.imshow(im_resized)
      plt.show()

search_text = "arrow"
results = 5
Search(search_text, results)