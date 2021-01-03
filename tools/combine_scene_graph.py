import os
import argparse

from tqdm import tqdm
import pickle
import json
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary_file', type=str, help='dictionary file')
    parser.add_argument('--word_matrix_file', type=str, help='word embedding matrix file')
    parser.add_argument('--sample_root_path', type=str, help='file path save the samples')
    parser.add_argument('--ids_map_folder', type=str, help='image index map file')
    parser.add_argument('--scene_graph_folder', type=str, help='file path save the scene graph')
    parser.add_argument('--export_folder', type=str, help='path to save result')
    return parser.parse_args()


class WordEmbeding:
    def __int__(self):
        self.word_to_idx = None
        self.index_to_vector = None

    def word_index_init(self, dictionary_file):
        with open(dictionary_file, 'r') as f:
            self.word_to_idx = json.load(f)['word_to_ix']

    def word_embedding_matrix_init(self, word_matrix_file):
        self.index_to_vector = np.load(word_matrix_file)

    def word_embedding(self, token):
        try:
            index = self.word_to_idx[token]
        except KeyError:
            index = self.word_to_idx['unknown']
        vector = self.index_to_vector[index]
        return vector


def scene_graph_embed(scene_graph, word_embed):
    scene_graph_embedding = np.zeros((36, 900), dtype='float32')

    for object_index, _object in enumerate(scene_graph['objects'].values()):
        # object embedding
        name_embedding = word_embed(_object['name'])

        # attribute embedding
        if len(_object['attributes']):
            attribute_embedding = np.zeros((len(_object['attributes']), 300), dtype='float32')
            for attribute_idx, attribute_name in enumerate(_object['attributes']):
                attribute_embedding[attribute_idx] = word_embed(attribute_name)
            attribute_embedding = np.mean(attribute_embedding, 0)
        else:
            attribute_embedding = np.zeros((1, 300), dtype='float32')

        # relation embedding
        relation_embedding = np.zeros((36, 300), dtype='float32')
        for relation_index, relation in enumerate(_object['relations']):
            relation_name = relation['name']
            words = relation_name.split()
            word_embeddings = np.zeros((len(words), 300), dtype='float32')
            for idx, word in enumerate(words):
                word_embeddings[idx] = word_embed(word)
            relation_name_embedding = np.mean(word_embeddings, axis=0)
            subject_emb = word_embed(scene_graph['objects'][str(relation['object'])]['name'])
            relation_embedding[relation_index] = (relation_name_embedding + subject_emb) / 2
        relation_embedding = np.mean(relation_embedding, axis=0)

        # final scene graph embedding
        scene_graph_embedding[object_index] = np.concatenate((name_embedding,
                                                              attribute_embedding,
                                                              relation_embedding), axis=None)

    return torch.from_numpy(scene_graph_embedding)


def add_scene_graph_in_tier(tier, word_embed, args):
    print('{} sample scene graph combination'.format(tier))
    with open(os.path.join(args.scene_graph_folder, 'textvqa_{}_scenes.json'.format(tier), 'r')) as f:
        scene_graphs = json.load(f)
    with open(os.path.join(args.ids_map_folder, '{}_ids_map.json'.format(tier), 'r')) as f:
        image_id_to_ix = json.load(f)['image_id_to_ix']

    sample_data_set = os.path.join(args.sample_root_path, tier)
    save_dir = os.path.join(args.export_folder, tier)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for sample_file in tqdm(os.listdir(sample_data_set), unit='sample', desc='adding scene graph feature to sample'):
        with open(os.path.join(sample_data_set, sample_file), 'rb') as f:
            sample = pickle.load(f)
        image_id = sample['image_id']
        scene_graph_index = image_id_to_ix[image_id]
        scene_graph = scene_graphs[scene_graph_index]
        sample['scene_graph'] = scene_graph_embed(scene_graph, word_embed)

        with open(os.path.join(save_dir, sample_file), "wb") as f:
            pickle.dump(sample, f)


def main():
    args = get_args()

    word_embed = WordEmbeding()
    word_embed.word_index_init(args.dictionary_file)
    word_embed.word_embedding_matrix_init(args.word_matrix_file)

    for tier in ['trian', 'val', 'test']:
        add_scene_graph_in_tier(tier, word_embed, args)


if __name__ == '__main__':
    main()

