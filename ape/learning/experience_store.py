import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import hashlib
from dataclasses import dataclass, asdict


@dataclass
class PhysicsExperience:
    """A single collision experience"""
    scenario_id: str
    timestamp: float
    agent_id: str
    
    my_velocity: List[float]
    my_mass: float
    my_elasticity: float
    other_velocity: List[float]
    other_mass: float
    other_elasticity: float
    collision_normal: List[float]
    
    predicted_my_velocity: List[float]
    predicted_other_velocity: List[float]
    reasoning: str
    
    actual_my_velocity: List[float]
    actual_other_velocity: List[float]
    
    prediction_error: float
    momentum_error: float
    energy_error: float
    was_correct: bool
    
    similar_experiences_used: int
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['was_correct'] = bool(data['was_correct'])
        data['prediction_error'] = float(data['prediction_error'])
        data['momentum_error'] = float(data['momentum_error'])
        data['energy_error'] = float(data['energy_error'])
        data['timestamp'] = float(data['timestamp'])
        data['my_mass'] = float(data['my_mass'])
        data['my_elasticity'] = float(data['my_elasticity'])
        data['other_mass'] = float(data['other_mass'])
        data['other_elasticity'] = float(data['other_elasticity'])
        data['similar_experiences_used'] = int(data['similar_experiences_used'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsExperience':
        return cls(**data)
    
    def to_text_description(self) -> str:
        """Convert to natural language for embedding"""
        return f"""
Physics collision scenario:
- Agent mass {self.my_mass}kg moving at velocity {self.my_velocity} m/s
- Collides with object mass {self.other_mass}kg moving at {self.other_velocity} m/s
- Collision normal: {self.collision_normal}
- Elasticity: mine={self.my_elasticity}, other={self.other_elasticity}
- Predicted outcome: my velocity {self.predicted_my_velocity}, other {self.predicted_other_velocity}
- Actual outcome: my velocity {self.actual_my_velocity}, other {self.actual_other_velocity}
- Error magnitude: {self.prediction_error:.3f}
- Agent reasoning: {self.reasoning}
"""


class ExperienceStore:
    """
    Vector database for storing and retrieving physics experiences
    
    Uses Qdrant for vector storage and sentence-transformers for embeddings
    """
    
    def __init__(
        self,
        collection_name: str = "physics_experiences",
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_path: str = "./qdrant_storage"
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create Qdrant collection for experiences"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"[ExperienceStore] Created collection: {self.collection_name}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        return self.encoder.encode(text).tolist()
    
    def store_experience(self, experience: PhysicsExperience) -> str:
        """
        Store a physics experience in vector database
        
        Returns the ID of the stored experience
        """
        text = experience.to_text_description()
        embedding = self._generate_embedding(text)
        
        exp_id = hashlib.md5(
            f"{experience.agent_id}_{experience.timestamp}".encode()
        ).hexdigest()
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=exp_id,
                    vector=embedding,
                    payload=experience.to_dict()
                )
            ]
        )
        
        return exp_id
    
    def retrieve_similar_experiences(
        self,
        query_scenario: Dict[str, Any],
        limit: int = 5,
        min_score: float = 0.7,
        agent_id: Optional[str] = None
    ) -> List[PhysicsExperience]:
        """
        Retrieve similar past experiences for a given scenario
        
        Args:
            query_scenario: Current scenario to match against
            limit: Maximum number of experiences to return
            min_score: Minimum similarity score (0-1)
            agent_id: Optional filter by specific agent
        
        Returns:
            List of similar experiences, ordered by similarity
        """
        query_text = f"""
Physics collision scenario:
- Agent mass {query_scenario['my_mass']}kg moving at velocity {query_scenario['my_velocity']} m/s
- Collides with object mass {query_scenario['other_mass']}kg moving at {query_scenario['other_velocity']} m/s
- Collision normal: {query_scenario['collision_normal']}
- Elasticity: mine={query_scenario['my_elasticity']}, other={query_scenario['other_elasticity']}
"""
        
        query_embedding = self._generate_embedding(query_text)
        
        query_filter = None
        if agent_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="agent_id",
                        match=MatchValue(value=agent_id)
                    )
                ]
            )
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=min_score,
            query_filter=query_filter
        ).points
        
        experiences = []
        for result in results:
            exp = PhysicsExperience.from_dict(result.payload)
            experiences.append(exp)
        
        return experiences
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """
        Get learning statistics for a specific agent
        
        Returns metrics like total experiences, accuracy over time, etc.
        """
        experiences = []
        offset = None
        
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="agent_id",
                    match=MatchValue(value=agent_id)
                )
            ]
        )
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset
            )
            
            for point in results:
                experiences.append(PhysicsExperience.from_dict(point.payload))
            
            if offset is None:
                break
        
        if not experiences:
            return {
                'total_experiences': 0,
                'accuracy': 0.0,
                'avg_error': 0.0
            }
        
        total = len(experiences)
        correct = sum(1 for exp in experiences if exp.was_correct)
        total_error = sum(exp.prediction_error for exp in experiences)
        
        experiences.sort(key=lambda x: x.timestamp)
        
        window_size = min(10, total)
        recent_experiences = experiences[-window_size:]
        recent_accuracy = sum(1 for exp in recent_experiences if exp.was_correct) / len(recent_experiences)
        
        return {
            'agent_id': agent_id,
            'total_experiences': total,
            'overall_accuracy': correct / total,
            'recent_accuracy': recent_accuracy,
            'avg_prediction_error': total_error / total,
            'improvement': recent_accuracy - (correct / total) if total > window_size else 0.0,
            'experiences_over_time': [
                {
                    'timestamp': exp.timestamp,
                    'was_correct': exp.was_correct,
                    'error': exp.prediction_error
                }
                for exp in experiences
            ]
        }
    
    def get_best_experiences(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[PhysicsExperience]:
        """Get the most accurate predictions (lowest error)"""
        all_experiences = []
        offset = None
        
        query_filter = None
        if agent_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="agent_id",
                        match=MatchValue(value=agent_id)
                    )
                ]
            )
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset
            )
            
            for point in results:
                all_experiences.append(PhysicsExperience.from_dict(point.payload))
            
            if offset is None:
                break
        
        all_experiences.sort(key=lambda x: x.prediction_error)
        return all_experiences[:limit]
    
    def get_worst_experiences(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[PhysicsExperience]:
        """Get the least accurate predictions (highest error)"""
        all_experiences = []
        offset = None
        
        query_filter = None
        if agent_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="agent_id",
                        match=MatchValue(value=agent_id)
                    )
                ]
            )
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset
            )
            
            for point in results:
                all_experiences.append(PhysicsExperience.from_dict(point.payload))
            
            if offset is None:
                break
        
        all_experiences.sort(key=lambda x: x.prediction_error, reverse=True)
        return all_experiences[:limit]
    
    def clear_collection(self):
        """Delete all experiences (for testing)"""
        self.client.delete_collection(collection_name=self.collection_name)
        self._initialize_collection()
