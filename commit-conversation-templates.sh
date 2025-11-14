#!/bin/bash

# Commit script for conversation templates and inter-turn delays

echo "============================================="
echo "Committing: Conversation Templates & Inter-Turn Delays"
echo "============================================="
echo ""

# Show what will be committed
echo "Modified files:"
git diff --name-only
echo ""

# Stage all changes
git add -A

# Show staged changes summary
echo "Staged changes:"
git diff --cached --stat
echo ""

# Create detailed commit message
git commit -m "feat: Add conversation templates and inter-turn delays

Implement realistic conversation patterns with natural think time pauses
between turns for production-like simulation behavior.

Conversation Templates:
----------------------
✅ Quick Q&A: 1-2 turns, 20-50 user tokens, 1-3s think time
✅ Standard Chat: 4-8 turns, 40-80 user tokens, 3-8s think time  
✅ Deep Dive: 10-20 turns, 80-200 user tokens, 5-15s think time
✅ Debug Session: 15-30 turns, 50-150 user tokens, 2-10s think time
✅ Mixed: Random selection from above templates

Inter-Turn Delays:
-----------------
- Simulates natural user think time between turns in same conversation
- Different delays for different conversation types
- Async sleep implementation (non-blocking)
- Verbose logging shows pause duration
- Request rate still controls conversation-level delays

Core Changes:
------------
src/multi_user_simulator.py:
  - Add ConversationTemplate enum with 5 template types
  - Add TemplateConfig dataclass with template parameters
  - Define TEMPLATE_CONFIGS dictionary with template specs
  - Add template and inter_turn_delay fields to Conversation
  - Implement generate_conversation_from_template() function
  - Add inter-turn delay logic in client_worker after each turn
  - Update generator creation to support template-based generation
  - Verbose output shows think time pauses

web/app.py:
  - Add conversation_template field to SimulationConfig
  - Parse template selection from user input
  - Generate initial conversations using templates
  - Pass template to continuous generation mode
  - Log selected template in event stream

web/static/index.html:
  - Add conversation template dropdown selector
  - Show template descriptions with turn/delay ranges
  - Default to \"Standard Chat\" template

web/static/app.js:
  - Read conversation_template from form
  - Pass to simulation API

README.md:
  - Document conversation templates feature
  - Document inter-turn delays feature
  - List all template types and characteristics

Benefits:
---------
✅ More realistic simulation behavior
✅ Natural conversation pacing
✅ Different user archetypes (quick vs deep discussions)
✅ Accurate cache warm-up patterns
✅ Production-like traffic patterns
✅ Better stress testing with varied workloads

Technical Details:
-----------------
- Inter-turn delays use asyncio.sleep() (non-blocking)
- Delays only between turns in SAME conversation
- Request rate controls delays BETWEEN conversations
- Templates define ranges for all parameters
- Mixed template randomly selects for each conversation
- Delays configurable per conversation instance
- Works with both text mode and synthetic mode

Example Behavior:
----------------
Standard Chat conversation:
  Turn 0: Process user message → get assistant response
  [Pause 5 seconds - user thinks]
  Turn 1: Process next user message → get assistant response  
  [Pause 7 seconds - user thinks]
  Turn 2: ...and so on

Use Cases:
----------
- Realistic user behavior simulation
- Long-running cache warm-up tests
- Multi-tenant workload analysis
- Performance testing with varied patterns
- Cache hit rate optimization studies"

echo ""
echo "============================================="
echo "Commit created successfully!"
echo "============================================="
echo ""
echo "Recent commits:"
git log --oneline -3
echo ""
echo "Remember to push when ready:"
echo "  git push origin <branch>"

