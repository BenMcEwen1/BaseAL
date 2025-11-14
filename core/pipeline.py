from typing import List, Tuple, Dict, Optional
from pathlib import Path

from .model import EmbeddingClassifier

# The parameters will be passed from the yaml file

class Pipeline:
    def __init__(
        self,
        embeddings_dir: Path,
        annotations_path: Path,
        model_name: str = "birdnet",
        dataset_name: str = "FewShot",
        hidden_dim: Optional[int] = None,
        learning_rate: float = 0.0001,
        device: str = "cpu"
    ):
        self.embeddings_dir = embeddings_dir
        self.annotations_path = annotations_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.device = device

        # Load data
        self.embeddings, self.labels, self.label_to_idx, self.idx_to_label = self._load_data()

        # Initialize model
        input_dim = self.embeddings.shape[1]
        num_classes = len(self.label_to_idx)
        self.model = EmbeddingClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes).to(self.device)
        
    def _load_data():
        pass
        
    def _generate_embeddings():
        pass

    def _train():
        pass
        
    def run_al_cycles(n_cycles):
        for i in range(n_cycles):
            # Do expensive computation
            embeddings = _generate_embeddings()  # Takes time
            results = _train_model()              # Takes time
            
            yield {
                'cycle': i,
                'embeddings': embeddings,
                'results': results
            }