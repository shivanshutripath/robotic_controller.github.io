#!/bin/bash
# Quick test to see if code_agent.py actually works with Claude

echo "Testing code_agent.py with Claude..."
echo ""

# Test 1: Check if code_agent has ModelClient
echo "Test 1: Checking for ModelClient class..."
if grep -q "class ModelClient" code_agent.py; then
    echo "✓ ModelClient found in code_agent.py"
else
    echo "❌ ModelClient NOT found - file not updated!"
    exit 1
fi

# Test 2: Check Python can import it
echo ""
echo "Test 2: Testing Python import..."
python3 -c "
import sys
import code_agent
if hasattr(code_agent, 'ModelClient'):
    print('✓ ModelClient can be imported')
    mc = code_agent.ModelClient
    if 'claude-sonnet-4.5' in mc.MODEL_MAP:
        print('✓ Claude models in MODEL_MAP')
    else:
        print('❌ Claude models NOT in MODEL_MAP')
        sys.exit(1)
else:
    print('❌ ModelClient not found after import')
    sys.exit(1)
" || exit 1

# Test 3: Check API key
echo ""
echo "Test 3: Checking ANTHROPIC_API_KEY..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY not set!"
    echo "   Run: export ANTHROPIC_API_KEY='your-key'"
    exit 1
else
    echo "✓ ANTHROPIC_API_KEY is set"
fi

# Test 4: Check anthropic package
echo ""
echo "Test 4: Checking anthropic package..."
python3 -c "import anthropic; print('✓ anthropic package installed')" || {
    echo "❌ anthropic not installed"
    echo "   Run: pip install anthropic"
    exit 1
}

# Test 5: Check required files exist
echo ""
echo "Test 5: Checking required files..."
REQUIRED_FILES=(
    "controller_template.py"
    "map_agent_outputs/occupancy.png"
    "map_agent_outputs/params.json"
    "DDR.png"
    "robot.py"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "✓ $f exists"
    else
        echo "❌ $f NOT FOUND"
        exit 1
    fi
done

# Test 6: Try to create a client
echo ""
echo "Test 6: Creating ModelClient for claude-sonnet-4.5..."
python3 -c "
from code_agent import ModelClient
try:
    client = ModelClient('claude-sonnet-4.5')
    print('✓ ModelClient created successfully')
    print(f'  Model: {client.model_name}')
    print(f'  API Model: {client.api_model}')
    print(f'  Is Claude: {client.is_claude}')
except Exception as e:
    print(f'❌ Failed to create client: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || exit 1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✓ All tests passed!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Now try running code_agent.py directly:"
echo ""
echo "python code_agent.py \\"
echo "  --template controller_template.py \\"
echo "  --map ./map_agent_outputs/occupancy.png \\"
echo "  --params ./map_agent_outputs/params.json \\"
echo "  --robot DDR.png --robotpy robot.py \\"
echo "  --out test_controller.py \\"
echo "  --model claude-sonnet-4.5 \\"
echo "  --debug"
echo ""