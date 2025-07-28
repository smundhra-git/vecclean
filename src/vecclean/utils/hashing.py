"""
Hashing utilities for VecClean.

Provides consistent hashing functions for content deduplication,
caching, integrity verification, and distributed systems support.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Any, Dict, List, Union, Iterator, Optional, Set, Tuple, Callable
from collections import defaultdict
import random
import bisect

try:
    import xxhash
    _XXHASH_AVAILABLE = True
except ImportError:
    _XXHASH_AVAILABLE = False


def generate_content_hash(
    content: str, 
    algorithm: str = "xxhash",
    encoding: str = "utf-8"
) -> str:
    """
    Generate a hash for content deduplication.
    
    Args:
        content: Text content to hash
        algorithm: Hash algorithm to use
        encoding: Text encoding to use
        
    Returns:
        Hexadecimal hash string
    """
    data = content.encode(encoding)
    
    if algorithm == "xxhash" and _XXHASH_AVAILABLE:
        return xxhash.xxh64(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    else:
        # Fallback to SHA-256
        return hashlib.sha256(data).hexdigest()


class LSHHasher:
    """
    Locality-Sensitive Hashing for similarity detection.
    
    Implements MinHash and SimHash algorithms for finding similar content
    without exact matching.
    """
    
    def __init__(self, num_hashes: int = 128, shingle_size: int = 3):
        """
        Initialize LSH hasher.
        
        Args:
            num_hashes: Number of hash functions to use
            shingle_size: Size of text shingles for hashing
        """
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self._hash_functions = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[Callable[[str], int]]:
        """Generate random hash functions for MinHash."""
        hash_funcs = []
        for i in range(self.num_hashes):
            a = random.randint(1, 2**32 - 1)
            b = random.randint(0, 2**32 - 1)
            hash_funcs.append(lambda x, a=a, b=b: (a * hash(x) + b) % (2**32))
        return hash_funcs
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Extract shingles from text."""
        shingles = set()
        words = text.lower().split()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            shingles.add(shingle)
        return shingles
    
    def minhash_signature(self, text: str) -> List[int]:
        """
        Generate MinHash signature for text.
        
        Args:
            text: Input text to hash
            
        Returns:
            MinHash signature as list of integers
        """
        shingles = self._get_shingles(text)
        signature = []
        
        for hash_func in self._hash_functions:
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = hash_func(shingle)
                min_hash = min(min_hash, hash_val)
            signature.append(min_hash if min_hash != float('inf') else 0)
        
        return signature
    
    def simhash_signature(self, text: str) -> int:
        """
        Generate SimHash signature for text.
        
        Args:
            text: Input text to hash
            
        Returns:
            SimHash signature as integer
        """
        shingles = self._get_shingles(text)
        vector = [0] * 64  # 64-bit hash
        
        for shingle in shingles:
            hash_val = hash(shingle)
            for i in range(64):
                bit = (hash_val >> i) & 1
                if bit:
                    vector[i] += 1
                else:
                    vector[i] -= 1
        
        # Convert to binary signature
        signature = 0
        for i in range(64):
            if vector[i] > 0:
                signature |= (1 << i)
        
        return signature
    
    def jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity from MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def hamming_distance(self, sig1: int, sig2: int) -> int:
        """Calculate Hamming distance between SimHash signatures."""
        return bin(sig1 ^ sig2).count('1')


class StreamingHasher:
    """
    Streaming hasher for processing large content incrementally.
    
    Allows hashing of content that doesn't fit in memory by processing
    it in chunks.
    """
    
    def __init__(self, algorithm: str = "xxhash"):
        """
        Initialize streaming hasher.
        
        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm
        self.reset()
    
    def reset(self):
        """Reset hasher state for new content."""
        if self.algorithm == "xxhash" and _XXHASH_AVAILABLE:
            self._hasher = xxhash.xxh64()
        elif self.algorithm == "sha256":
            self._hasher = hashlib.sha256()
        elif self.algorithm == "md5":
            self._hasher = hashlib.md5()
        else:
            self._hasher = hashlib.sha256()
    
    def update(self, chunk: Union[str, bytes]):
        """
        Update hash with new chunk of data.
        
        Args:
            chunk: Data chunk to add to hash
        """
        if isinstance(chunk, str):
            chunk = chunk.encode('utf-8')
        self._hasher.update(chunk)
    
    def digest(self) -> str:
        """Get final hash digest."""
        return self._hasher.hexdigest()


class ConsistentHashRing:
    """
    Consistent hashing ring for distributed systems.
    
    Provides even distribution of keys across nodes with minimal
    redistribution when nodes are added or removed.
    """
    
    def __init__(self, nodes: Optional[List[str]] = None, replicas: int = 100):
        """
        Initialize consistent hash ring.
        
        Args:
            nodes: Initial list of nodes
            replicas: Number of virtual nodes per physical node
        """
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def _hash(self, key: str) -> int:
        """Hash function for ring positions."""
        return int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    
    def add_node(self, node: str):
        """Add a node to the ring."""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            key = self._hash(virtual_key)
            self.ring[key] = node
        self._update_sorted_keys()
    
    def remove_node(self, node: str):
        """Remove a node from the ring."""
        for i in range(self.replicas):
            virtual_key = f"{node}:{i}"
            key = self._hash(virtual_key)
            if key in self.ring:
                del self.ring[key]
        self._update_sorted_keys()
    
    def _update_sorted_keys(self):
        """Update sorted list of ring keys."""
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a given key."""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first node clockwise from the hash
        for ring_key in self.sorted_keys:
            if hash_key <= ring_key:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def get_nodes(self, key: str, count: int = 1) -> List[str]:
        """Get multiple nodes for replication."""
        if not self.ring or count <= 0:
            return []
        
        hash_key = self._hash(key)
        nodes = []
        seen_nodes = set()
        
        # Find starting position
        start_idx = 0
        for i, ring_key in enumerate(self.sorted_keys):
            if hash_key <= ring_key:
                start_idx = i
                break
        
        # Collect unique nodes
        idx = start_idx
        while len(nodes) < count and len(seen_nodes) < len(set(self.ring.values())):
            node = self.ring[self.sorted_keys[idx]]
            if node not in seen_nodes:
                nodes.append(node)
                seen_nodes.add(node)
            idx = (idx + 1) % len(self.sorted_keys)
        
        return nodes


class HashVerifier:
    """
    Hash verification and collision detection utilities.
    
    Provides tools for verifying content integrity and detecting
    hash collisions in large datasets.
    """
    
    def __init__(self):
        """Initialize hash verifier."""
        self.collision_tracker = defaultdict(list)
        self.verification_cache = {}
    
    def verify_integrity(
        self, 
        content: str, 
        expected_hash: str, 
        algorithm: str = "xxhash"
    ) -> bool:
        """
        Verify content integrity against expected hash.
        
        Args:
            content: Content to verify
            expected_hash: Expected hash value
            algorithm: Hash algorithm to use
            
        Returns:
            True if content matches expected hash
        """
        actual_hash = generate_content_hash(content, algorithm)
        return actual_hash == expected_hash
    
    def detect_collision(self, content: str, hash_value: str) -> bool:
        """
        Detect potential hash collisions.
        
        Args:
            content: Content to check
            hash_value: Hash value of content
            
        Returns:
            True if collision detected
        """
        if hash_value in self.collision_tracker:
            for existing_content in self.collision_tracker[hash_value]:
                if existing_content != content:
                    return True
        
        self.collision_tracker[hash_value].append(content)
        return False
    
    def get_collision_stats(self) -> Dict[str, Any]:
        """Get collision detection statistics."""
        total_hashes = len(self.collision_tracker)
        collisions = sum(1 for contents in self.collision_tracker.values() 
                        if len(contents) > 1)
        
        return {
            "total_hashes": total_hashes,
            "collisions": collisions,
            "collision_rate": collisions / total_hashes if total_hashes > 0 else 0,
            "max_collision_count": max(len(contents) for contents in self.collision_tracker.values()) if self.collision_tracker else 0
        }
    
    def clear_cache(self):
        """Clear verification cache and collision tracker."""
        self.collision_tracker.clear()
        self.verification_cache.clear()


# Global instances for common use
_lsh_hasher = LSHHasher()
_streaming_hasher = StreamingHasher()
_hash_verifier = HashVerifier()


def compute_similarity_hash(text: str, method: str = "minhash") -> Union[List[int], int]:
    """
    Compute similarity hash for text.
    
    Args:
        text: Input text
        method: Similarity method ('minhash' or 'simhash')
        
    Returns:
        Similarity hash signature
    """
    if method == "minhash":
        return _lsh_hasher.minhash_signature(text)
    elif method == "simhash":
        return _lsh_hasher.simhash_signature(text)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def stream_hash_file(file_path: str, algorithm: str = "xxhash") -> str:
    """
    Hash a file using streaming approach.
    
    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm to use
        
    Returns:
        File hash
    """
    hasher = StreamingHasher(algorithm)
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return hasher.digest()


def create_consistent_ring(nodes: List[str], replicas: int = 100) -> ConsistentHashRing:
    """
    Create a consistent hash ring.
    
    Args:
        nodes: List of node identifiers
        replicas: Number of virtual nodes per physical node
        
    Returns:
        Configured hash ring
    """
    return ConsistentHashRing(nodes, replicas)


def verify_content_integrity(content: str, expected_hash: str, algorithm: str = "xxhash") -> bool:
    """
    Verify content integrity.
    
    Args:
        content: Content to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm to use
        
    Returns:
        True if content is valid
    """
    return _hash_verifier.verify_integrity(content, expected_hash, algorithm)


__all__ = [
    "generate_content_hash",
    "LSHHasher", 
    "StreamingHasher",
    "ConsistentHashRing",
    "HashVerifier",
    "compute_similarity_hash",
    "stream_hash_file",
    "create_consistent_ring",
    "verify_content_integrity",
] 