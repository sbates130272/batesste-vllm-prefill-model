#!/bin/bash

# Commit script for continuous generation mode with time/turn limits

echo "============================================="
echo "Committing: Continuous Generation Mode"
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
git commit -m "feat: Add continuous generation mode with time and turn limits

Implement continuous conversation generation with flexible duration 
controls for long-running simulations and realistic workload testing.

Core Changes:
------------
- Add generate_new_conversation() function for on-the-fly conversation 
  creation
- Modify client_worker() to accept conversation_generator callback
- Update run_multi_user_simulation() to create and pass generators to 
  clients
- Conversations now regenerate continuously when originals complete

Web UI Changes:
--------------
- Add 'Time Limit' input field (default: 60s, -1 for infinite)
- Add 'Max Total Turns' input field (0 for unlimited)
- Update backend to track both time and turn limits
- Add early termination logic when limits are reached
- Change default request rate from 0 to 1.0 req/sec for better UX

Documentation:
-------------
- Update README.md with continuous generation features
- Add 'Simulation Duration Controls' section to WEB_UI.md
- Document time limit behavior (-1 = infinite, ≥0 = seconds)
- Document turn limit behavior (0 = unlimited, >0 = max turns)
- Provide usage examples for different simulation scenarios

Features:
---------
✅ Continuous generation - new conversations auto-created on completion
✅ Time-based limits - stop after N seconds (-1 for infinite)
✅ Turn-based limits - stop after N total turns (0 for unlimited)
✅ Dual limit support - stops at first limit reached
✅ Better defaults - 1.0 req/sec rate, 60s time limit

Use Cases:
----------
- Long-running performance tests
- Continuous workload simulation
- Cache behavior analysis over time
- Production-like traffic patterns
- Quick iteration testing (60s default)

Technical Notes:
---------------
- Generator functions use closures to maintain per-client state
- Token IDs use client_id and counter for uniqueness
- Generators created per-client with independent counters
- Time tracking starts on visualizer initialization
- Early stop checked after each turn completion

Example Configurations:
----------------------
1. Quick test: max_total_turns=100, time_limit=60
2. Long run: max_total_turns=0, time_limit=300
3. Infinite: max_total_turns=0, time_limit=-1
4. Turn-based: max_total_turns=1000, time_limit=-1"

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

