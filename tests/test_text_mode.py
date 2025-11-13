"""Test text mode functionality."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import text_mode


def test_text_message_creation():
    """Test TextMessage can be created."""
    msg = text_mode.TextMessage(
        role="user",
        content="Hello, world!",
        token_ids=[1, 2, 3, 4]
    )
    assert msg.role == "user"
    assert msg.content == "Hello, world!"
    assert msg.token_ids == [1, 2, 3, 4]


def test_text_conversation_creation():
    """Test TextConversation can be created."""
    messages = [
        text_mode.TextMessage(role="user", content="Hi", token_ids=[1, 2]),
        text_mode.TextMessage(role="assistant", content="Hello!", token_ids=[3, 4, 5])
    ]
    
    conv = text_mode.TextConversation(
        conv_id="test_conv",
        messages=messages
    )
    
    assert conv.conv_id == "test_conv"
    assert len(conv.messages) == 2
    assert conv.current_turn == 0


def test_text_conversation_get_messages():
    """Test getting messages up to a turn."""
    messages = [
        text_mode.TextMessage(role="user", content="Hi", token_ids=[1, 2]),
        text_mode.TextMessage(role="assistant", content="Hello!", token_ids=[3, 4, 5]),
        text_mode.TextMessage(role="user", content="How are you?", token_ids=[6, 7, 8, 9])
    ]
    
    conv = text_mode.TextConversation(conv_id="test", messages=messages)
    
    # Get tokens up to turn 0 (just first message)
    tokens0 = conv.get_messages_up_to_turn(0)
    assert tokens0 == [1, 2]
    
    # Get tokens up to turn 1 (first two messages)
    tokens1 = conv.get_messages_up_to_turn(1)
    assert tokens1 == [1, 2, 3, 4, 5]
    
    # Get tokens up to turn 2 (all three messages)
    tokens2 = conv.get_messages_up_to_turn(2)
    assert tokens2 == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_text_conversation_has_more_turns():
    """Test has_more_turns()."""
    messages = [
        text_mode.TextMessage(role="user", content="Hi", token_ids=[1, 2]),
        text_mode.TextMessage(role="assistant", content="Hello!", token_ids=[3, 4, 5])
    ]
    
    conv = text_mode.TextConversation(conv_id="test", messages=messages)
    conv.current_turn = 0
    
    assert conv.has_more_turns()  # turn 0, still has turn 1
    
    conv.current_turn = 1
    assert not conv.has_more_turns()  # turn 1 is last


def test_text_dataset_creation():
    """Test TextModeDataset can be created."""
    dataset = text_mode.TextModeDataset(tokenizer_name="gpt2")
    assert dataset.tokenizer_name == "gpt2"
    assert dataset.conversations == []


def test_synthetic_conversation_generation():
    """Test synthetic conversation generation."""
    dataset = text_mode.TextModeDataset()
    dataset.load_sharegpt_dataset(max_conversations=3)
    
    # Should have generated 3 conversations
    assert len(dataset.conversations) == 3
    
    # Each conversation should have messages
    for conv in dataset.conversations:
        assert 'id' in conv
        assert 'messages' in conv
        assert len(conv['messages']) > 0


def test_get_text_conversations():
    """Test converting to TextConversation objects."""
    dataset = text_mode.TextModeDataset()
    dataset.load_sharegpt_dataset(max_conversations=5)
    
    # Get text conversations
    convs = dataset.get_text_conversations(num_conversations=3)
    
    assert len(convs) == 3
    assert all(isinstance(c, text_mode.TextConversation) for c in convs)
    
    # Check structure
    conv = convs[0]
    assert hasattr(conv, 'conv_id')
    assert hasattr(conv, 'messages')
    assert len(conv.messages) > 0
    assert all(hasattr(m, 'content') for m in conv.messages)
    assert all(hasattr(m, 'token_ids') for m in conv.messages)


def test_dataset_stats():
    """Test dataset statistics."""
    dataset = text_mode.TextModeDataset()
    dataset.load_sharegpt_dataset(max_conversations=5)
    
    stats = dataset.get_stats()
    
    assert 'num_conversations' in stats
    assert stats['num_conversations'] == 5
    assert 'total_messages' in stats
    assert stats['total_messages'] > 0
    assert 'avg_turns_per_conv' in stats
    assert stats['avg_turns_per_conv'] > 0

