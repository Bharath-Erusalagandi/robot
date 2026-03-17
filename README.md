## Abluo

DishSpace is a kitchen-manipulation training and evaluation stack for robot arms that need to deal with real sink scenes: clutter, wet surfaces, transparent glass, reflective metal, stacked dishes, and dishwasher loading.

## Goal

The end goal is not a single demo grasp endpoint. The end goal is a robot policy that can:

1. Perceive almost all common sink items.
2. Select stable grasps in cluttered, wet scenes.
3. Decide what to pick first and where it should go.
4. Transfer objects to the drying rack or dishwasher.
5. Recover when grasping or placement fails.

## Current Scope

The current codebase is in the grasp-core phase:

1. Synthetic kitchen grasp data generation.
2. Vision-language-action dataset preparation.
3. Adapter fine-tuning scaffolding.
4. Grasp evaluation and API plumbing.

That is the right first phase, but it is not yet the full sink-to-dishwasher system.

## Development Phases

### Phase 1: Grasp Foundation

- Generate balanced synthetic data across kitchen object families.
- Train a grasp model that works in wet, cluttered sink scenes.
- Benchmark on sink-scene categories, not only isolated objects.

### Phase 2: Placement and Loading

- Add placement targets such as drying rack, utensil caddy, dishwasher top rack, and dishwasher bottom rack.
- Train policies for safe transfer and placement, not only pickup.

### Phase 3: Sequencing

- Learn task order: clear large blockers, separate utensils, protect fragile items, then load dishwasher.
- Add retry and recovery after failed grasps or collisions.

### Phase 4: Real-World Adaptation

- Fine-tune on pilot kitchen rollouts.
- Close the sim-to-real gap with real failures, retries, and placement logs.

## What Good Looks Like

- Thousands to tens of thousands of synthetic grasp attempts.
- Real cluttered sink scenes, not only isolated object crops.
- Both successful and failed attempts retained for evaluation and recovery modeling.
- Benchmark categories that stress clutter, occlusion, wetness, utensil entanglement, and placement.
