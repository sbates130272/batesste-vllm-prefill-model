"""
Text Mode for vLLM Prefill Simulator

Uses real text conversations with actual tokenization instead of
synthetic integer token IDs.

Supports ShareGPT dataset format with real tokenizers.
"""
import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TextMessage:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    token_ids: List[int]


@dataclass
class TextConversation:
    """A conversation with real text and tokenization."""
    conv_id: str
    messages: List[TextMessage]
    current_turn: int = 0
    
    def get_messages_up_to_turn(self, turn: int) -> List[int]:
        """
        Get all token IDs up to and including the specified turn.
        
        Returns concatenated token IDs from all messages up to turn.
        """
        tokens = []
        for i in range(min(turn + 1, len(self.messages))):
            tokens.extend(self.messages[i].token_ids)
        return tokens
    
    def has_more_turns(self) -> bool:
        """Check if there are more turns in this conversation."""
        return self.current_turn < len(self.messages) - 1


class TextModeDataset:
    """
    Manages text-mode conversations with real tokenization.
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        tokenizer = None
    ):
        """
        Initialize text mode dataset.
        
        Args:
            dataset_path: Path to ShareGPT JSON file (optional)
            tokenizer_name: HuggingFace tokenizer to use
            tokenizer: Pre-loaded tokenizer object (optional, takes precedence)
        """
        self.dataset_path = dataset_path
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer  # Use provided tokenizer if available
        self.conversations: List[Dict[str, Any]] = []
        
        # Try to load tokenizer if not provided
        if not self.tokenizer:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    trust_remote_code=True
                )
                print(f"✓ Loaded tokenizer: {tokenizer_name}")
            except ImportError:
                print("WARNING: transformers not installed. "
                      "Install with: pip install transformers")
                print("Falling back to mock tokenization...")
            except Exception as e:
                print(f"WARNING: Could not load tokenizer: {e}")
                print("Falling back to mock tokenization...")
        else:
            print(f"✓ Using provided tokenizer")
    
    def load_sharegpt_dataset(
        self, 
        max_conversations: int = 100,
        min_turns: int = 2,
        max_turns: int = 20
    ):
        """
        Load ShareGPT format dataset.
        
        Supports multiple ShareGPT format variations:
        
        Format 1 (Standard):
        [
            {
                "id": "conv_id",
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]
            },
            ...
        ]
        
        Format 2 (Vicuna):
        [
            {
                "id": "conv_id",
                "conversations": [
                    {"from": "human", "value": "..."},
                    {"from": "gpt", "value": "..."},
                    ...
                ]
            },
            ...
        ]
        
        Args:
            max_conversations: Maximum conversations to load
            min_turns: Minimum number of turns to include
            max_turns: Maximum number of turns to include
        """
        if not self.dataset_path:
            print("No dataset path provided. "
                  "Generating synthetic text...")
            self._generate_synthetic_text_conversations(max_conversations)
            return
        
        try:
            print(f"Loading ShareGPT dataset from: {self.dataset_path}")
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, list):
                raw_conversations = data
            elif isinstance(data, dict) and 'conversations' in data:
                raw_conversations = data['conversations']
            elif isinstance(data, dict) and 'data' in data:
                raw_conversations = data['data']
            else:
                raise ValueError(
                    "Unknown dataset format. "
                    "Expected list of conversations."
                )
            
            print(f"Loaded {len(raw_conversations):,} raw conversations")
            
            # Normalize and filter conversations
            normalized = self._normalize_conversations(
                raw_conversations,
                min_turns=min_turns,
                max_turns=max_turns
            )
            
            # Sample if needed
            if len(normalized) > max_conversations:
                import random
                normalized = random.sample(normalized, max_conversations)
            
            self.conversations = normalized
            print(f"✓ Using {len(self.conversations):,} conversations "
                  f"({min_turns}-{max_turns} turns)")
            
        except FileNotFoundError:
            print(f"Dataset file not found: {self.dataset_path}")
            print("Generating synthetic text conversations instead...")
            self._generate_synthetic_text_conversations(max_conversations)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Generating synthetic text conversations instead...")
            self._generate_synthetic_text_conversations(max_conversations)
    
    def _normalize_conversations(
        self,
        raw_conversations: List[Dict],
        min_turns: int = 2,
        max_turns: int = 20
    ) -> List[Dict]:
        """
        Normalize conversations to standard format and filter.
        
        Args:
            raw_conversations: Raw conversation data
            min_turns: Minimum turns to keep
            max_turns: Maximum turns to keep
        
        Returns:
            Normalized conversations in standard format
        """
        normalized = []
        
        for conv in raw_conversations:
            try:
                # Extract ID
                conv_id = conv.get('id', conv.get('conversation_id', 
                                   f'conv_{len(normalized):04d}'))
                
                # Extract messages (handle multiple field names)
                messages_raw = (conv.get('messages') or 
                               conv.get('conversations') or
                               conv.get('dialog') or [])
                
                if not messages_raw:
                    continue
                
                # Normalize message format
                messages = []
                for msg in messages_raw:
                    # Extract role (handle multiple field names)
                    role = (msg.get('role') or 
                           msg.get('from') or 
                           msg.get('sender'))
                    
                    # Normalize role names
                    if role in ['human', 'user', 'USER']:
                        role = 'user'
                    elif role in ['gpt', 'assistant', 'ASSISTANT', 
                                  'bot']:
                        role = 'assistant'
                    elif role == 'system':
                        # Skip system messages for now
                        continue
                    else:
                        # Unknown role, skip
                        continue
                    
                    # Extract content
                    content = (msg.get('content') or 
                              msg.get('value') or 
                              msg.get('text') or '')
                    
                    if not content.strip():
                        continue
                    
                    messages.append({
                        'role': role,
                        'content': content.strip()
                    })
                
                # Filter by number of turns
                num_messages = len(messages)
                if num_messages < min_turns * 2:
                    continue
                if num_messages > max_turns * 2:
                    # Truncate to max_turns
                    messages = messages[:max_turns * 2]
                
                # Ensure alternating user/assistant pattern
                if not self._validate_alternating(messages):
                    continue
                
                normalized.append({
                    'id': conv_id,
                    'messages': messages
                })
                
            except Exception as e:
                # Skip conversations that fail to parse
                continue
        
        return normalized
    
    def _validate_alternating(self, messages: List[Dict]) -> bool:
        """
        Check if messages alternate between user and assistant.
        
        Args:
            messages: List of message dicts
        
        Returns:
            True if valid alternating pattern
        """
        if not messages:
            return False
        
        # Should start with user
        if messages[0]['role'] != 'user':
            return False
        
        # Check alternating pattern
        for i in range(len(messages) - 1):
            curr_role = messages[i]['role']
            next_role = messages[i + 1]['role']
            if curr_role == next_role:
                return False
        
        return True
    
    def _generate_synthetic_text_conversations(
        self,
        num_conversations: int
    ):
        """
        Generate synthetic text conversations as fallback.
        
        Creates realistic-looking multi-turn dialogues with variety.
        """
        print(f"Generating {num_conversations} synthetic text conversations...")
        
        topics = [
            "machine learning", "neural networks", "data pipelines", "web APIs",
            "cloud infrastructure", "AI models", "SQL optimization", "authentication",
            "microservices", "mobile UI design", "CI/CD pipelines", "sorting algorithms",
            "distributed systems", "container orchestration", "cryptography", "caching strategies"
        ]
        
        first_turn_templates = [
            "Can you explain {topic} and how it works in production?",
            "I'm trying to understand {topic} - where should I start?",
            "What are the best practices for implementing {topic}?",
            "How does {topic} compare to alternative approaches?",
            "I'm debugging an issue with {topic}, can you help?",
            "What are the performance implications of {topic}?",
            "Could you walk me through {topic} with an example?",
            "When should I use {topic} vs other solutions?"
        ]
        
        follow_up_templates = [
            "That makes sense! What about the edge cases?",
            "Interesting. How does this scale in practice?",
            "Can you give me a concrete code example?",
            "What are the common pitfalls to avoid?",
            "How would this work in a distributed environment?",
            "What about security considerations?",
            "Could you elaborate on the performance aspects?",
            "Are there any alternatives I should consider?"
        ]
        
        assistant_intros = [
            "Great question! Let me explain how this works.",
            "Sure, I can help with that.",
            "Here's a comprehensive overview:",
            "Let me break this down for you.",
            "That's an important topic. Here's what you need to know:",
            "Good question - this is a common challenge.",
            "I'll explain the key concepts you need to understand."
        ]
        
        assistant_bodies = [
            "The core principle involves careful design and implementation. You'll want to consider both performance and maintainability.",
            "There are several approaches, each with tradeoffs. The most common pattern is to balance simplicity with flexibility.",
            "In production systems, you need to account for failure modes and recovery strategies. Testing is crucial here.",
            "The key is understanding the underlying architecture. Once you grasp that, implementation becomes straightforward.",
            "Best practices suggest starting simple and iterating based on metrics. Don't over-engineer early on.",
            "Modern implementations focus on scalability and resilience. You'll want to leverage existing libraries where possible.",
            "The implementation details depend on your specific use case, but the general principles remain consistent."
        ]
        
        for i in range(num_conversations):
            topic = random.choice(topics)
            num_turns = random.randint(2, 6)
            
            messages = []
            for turn in range(num_turns):
                # User message
                if turn == 0:
                    template = random.choice(first_turn_templates)
                    user_content = template.format(topic=topic)
                else:
                    user_content = random.choice(follow_up_templates)
                
                messages.append({
                    "role": "user",
                    "content": user_content
                })
                
                # Assistant message - more varied
                intro = random.choice(assistant_intros)
                body = random.choice(assistant_bodies)
                assistant_content = f"{intro} {body}"
                
                # Add occasional follow-up questions
                if random.random() < 0.3:
                    assistant_content += " Would you like me to elaborate on any specific part?"
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            self.conversations.append({
                "id": f"conv_{i:04d}",
                "messages": messages
            })
        
        print(f"✓ Generated {len(self.conversations)} conversations")
    
    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text using the loaded tokenizer or mock tokens.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of token IDs
        """
        if self.tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Mock tokenization: ~4 chars per token (rough estimate)
            num_tokens = max(1, len(text) // 4)
            # Generate sequential token IDs based on hash
            base = hash(text) % 10000
            return list(range(base, base + num_tokens))
    
    def get_text_conversations(
        self,
        num_conversations: int,
        common_prefix: Optional[str] = None
    ) -> List[TextConversation]:
        """
        Convert dataset conversations to TextConversation objects.
        
        Args:
            num_conversations: Number of conversations to return
            common_prefix: Optional common system prompt for all conversations
        
        Returns:
            List of TextConversation objects
        """
        text_convs = []
        
        # Sample conversations
        sampled = random.sample(
            self.conversations,
            min(num_conversations, len(self.conversations))
        )
        
        common_prefix_tokens = []
        if common_prefix:
            common_prefix_tokens = self._tokenize_text(common_prefix)
        
        for conv_data in sampled:
            conv_id = conv_data.get("id", f"conv_{len(text_convs):04d}")
            messages_data = conv_data.get("messages", [])
            
            text_messages = []
            for msg_data in messages_data:
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
                
                # Tokenize content
                token_ids = self._tokenize_text(content)
                
                # Add common prefix to first user message
                if role == "user" and len(text_messages) == 0 and common_prefix_tokens:
                    token_ids = common_prefix_tokens + token_ids
                
                text_messages.append(TextMessage(
                    role=role,
                    content=content,
                    token_ids=token_ids
                ))
            
            text_convs.append(TextConversation(
                conv_id=conv_id,
                messages=text_messages
            ))
        
        return text_convs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        if not self.conversations:
            return {}
        
        total_turns = []
        total_messages = 0
        total_chars = 0
        
        for conv in self.conversations:
            messages = conv.get("messages", [])
            total_turns.append(len(messages))
            total_messages += len(messages)
            for msg in messages:
                total_chars += len(msg.get("content", ""))
        
        return {
            "num_conversations": len(self.conversations),
            "total_messages": total_messages,
            "total_chars": total_chars,
            "avg_turns_per_conv": sum(total_turns) / len(total_turns) if total_turns else 0,
            "min_turns": min(total_turns) if total_turns else 0,
            "max_turns": max(total_turns) if total_turns else 0,
        }

