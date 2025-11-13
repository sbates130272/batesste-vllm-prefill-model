"""Test that all modules can be imported successfully."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_import_vllm_prefill_model():
    """Test that vllm-prefill-model module imports."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vllm_prefill_model",
        os.path.join(os.path.dirname(__file__), '..', 'src', 'vllm-prefill-model.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check key classes exist
    assert hasattr(module, 'KVBlock')
    assert hasattr(module, 'SimulatedRequest')
    assert hasattr(module, 'PrefixCacheManager')


def test_import_multi_user_simulator():
    """Test that multi_user_simulator module imports."""
    import multi_user_simulator
    
    # Check key functions exist
    assert hasattr(multi_user_simulator, 'generate_synthetic_conversations')
    assert hasattr(multi_user_simulator, 'run_multi_user_simulation')
    assert hasattr(multi_user_simulator, 'client_worker')


def test_import_text_mode():
    """Test that text_mode module imports."""
    import text_mode
    
    # Check key classes exist
    assert hasattr(text_mode, 'TextMessage')
    assert hasattr(text_mode, 'TextConversation')
    assert hasattr(text_mode, 'TextModeDataset')


def test_import_visualizer():
    """Test that visualizer module imports (without matplotlib)."""
    try:
        import visualizer
        # Check key classes exist
        assert hasattr(visualizer, 'SimulationVisualizer')
        assert hasattr(visualizer, 'NullVisualizer')
    except ImportError:
        # OK if matplotlib not available in CI
        pass

