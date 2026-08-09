import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

COLORS_MAP = {"red": "#e74c3c", "blue": "#3498db", "green": "#2ecc71", "yellow": "#f1c40f", "orange": "#e67e22"}


def sample_from_dict(prob_dict):
    """
    Sample an item from a dictionary of probabilities.

    Args:
        prob_dict: Dictionary mapping items to their probabilities.
                   Probabilities should sum to 1.

    Returns:
        A randomly sampled key from the dictionary, weighted by probabilities.

    Example:
        >>> sample_from_dict({"red": 0.4, "blue": 0.3, "green": 0.2, "yellow": 0.1})
        'red'  # (with 40% probability)
    """
    assert isinstance(prob_dict, dict), "Input must be a dictionary"
    assert len(prob_dict) > 0, "Input must not be empty"
    assert all(isinstance(k, str) for k in prob_dict.keys()), "Keys must be strings"
    for v in prob_dict.values():
        assert isinstance(v, float), "Probabilities must be floats"
        assert 0 <= v <= 1, "Probabilities must be between 0 and 1"
    assert abs(sum(prob_dict.values()) - 1) < 1e-6, "Probabilities must sum to 1"

    # Note: .keys() and .values() iterate in the same order (guaranteed in Python 3.7+)
    items = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return random.choices(items, weights=weights)[0]


def sample_from_list(items):
    """
    Sample uniformly from a list of items.

    Args:
        items: List of items to sample from.

    Returns:
        A randomly selected item from the list.

    Example:
        >>> sample_from_list(["apple", "banana", "banana", "mango"])
        'banana'  # (with 50% probability)
    """
    assert len(items) > 0, "Input must be non-empty"
    return random.choice(items)


def visualize(bag, sample_sizes):
    """
    Visualize how sampling converges to the true distribution.

    Args:
        bag: List of color strings (e.g., ["red", "red", "blue", ...])
        sample_sizes: List of sample sizes to demonstrate (e.g., [50, 200, 1000, 5000])
    """
    # Validate sample sizes
    if any(n < 0 for n in sample_sizes):
        raise ValueError("Sample sizes must be non-negative")
    # make sure all colors are in COLORS_MAP
    if not all(color in COLORS_MAP for color in set(bag)):
        raise ValueError(f"Supported colors are: {', '.join(COLORS_MAP.keys())}")

    def draw_cup_with_skittles(ax, proportions, n_skittles, title):
        """Draw a cup with naturally arranged Skittles"""
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, pad=10)

        # Draw cup shape (trapezoid)
        cup_bottom_width = 1.5
        cup_top_width = 2.5
        cup_height = 6
        cup_base = 0.5

        # Cup outline
        cup = patches.Polygon([
            (-cup_bottom_width / 2, cup_base),
            (cup_bottom_width / 2, cup_base),
            (cup_top_width / 2, cup_height),
            (-cup_top_width / 2, cup_height)
        ], fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(cup)

        # Create Skittles based on proportions
        skittles_by_color = []
        for color, prop in proportions.items():
            count = int(prop * n_skittles)
            skittles_by_color.extend([color] * count)

        # Shuffle for random arrangement
        random.shuffle(skittles_by_color)

        # Place Skittles in a natural "poured" pattern
        skittle_radius = 0.12
        placed_skittles = []

        # Create layers from bottom up
        y = cup_base + skittle_radius + 0.1

        while skittles_by_color and y < cup_height - skittle_radius:
            # Calculate cup width at this height
            height_ratio = (y - cup_base) / (cup_height - cup_base)
            current_width = cup_bottom_width + (cup_top_width - cup_bottom_width) * height_ratio

            # How many Skittles fit at this level?
            n_per_row = int((current_width - 2 * skittle_radius) / (2 * skittle_radius))

            # Place Skittles in this row with some randomness
            x_positions = np.linspace(-current_width / 2 + skittle_radius,
                                      current_width / 2 - skittle_radius,
                                      n_per_row)

            for x in x_positions:
                if not skittles_by_color:
                    break

                # Add some random jitter
                x_jitter = x + random.uniform(-0.03, 0.03)
                y_jitter = y + random.uniform(-0.02, 0.02)

                color = skittles_by_color.pop()
                circle = patches.Circle((x_jitter, y_jitter), skittle_radius,
                                        color=COLORS_MAP[color],
                                        ec='black', linewidth=0.5, alpha=0.9)
                ax.add_patch(circle)
                placed_skittles.append((x_jitter, y_jitter))

            # Move up for next layer
            y += 2 * skittle_radius * 0.85  # Slightly overlapping layers

    def draw_cup_as_stacked_bars(ax, proportions):
        """Draw a cup with color proportions as stacked rectangles"""
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw cup outline (trapezoid) - same dimensions as above
        cup_bottom_width = 1.5
        cup_top_width = 2.5
        cup_height = 6
        cup_base = 0.5

        cup = patches.Polygon([
            (-cup_bottom_width / 2, cup_base),
            (cup_bottom_width / 2, cup_base),
            (cup_top_width / 2, cup_height),
            (-cup_top_width / 2, cup_height)
        ], fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(cup)

        # Fill cup with colored sections proportional to the distribution
        colors = ["red", "blue", "green", "yellow"]
        y_current = cup_base
        total_height = cup_height - cup_base - 0.2  # Leave some space at top

        # Calculate percentages that sum to 100
        percentages = {}
        remaining = 100
        for i, color in enumerate(colors):
            if color in proportions and proportions[color] > 0:
                if i == len(colors) - 1:  # Last color gets the remainder
                    percentages[color] = remaining
                else:
                    percentages[color] = round(proportions[color] * 100)
                    remaining -= percentages[color]

        for color in colors:
            if color in proportions and proportions[color] > 0:
                # Height of this color section
                section_height = proportions[color] * total_height

                # Create points for this section (accounting for cup taper)
                y_top = y_current + section_height

                # Calculate widths at current and top positions
                ratio_bottom = (y_current - cup_base) / (cup_height - cup_base)
                ratio_top = (y_top - cup_base) / (cup_height - cup_base)

                width_bottom = cup_bottom_width + (cup_top_width - cup_bottom_width) * ratio_bottom
                width_top = cup_bottom_width + (cup_top_width - cup_bottom_width) * ratio_top

                # Create the section
                section = patches.Polygon([
                    (-width_bottom / 2, y_current),
                    (width_bottom / 2, y_current),
                    (width_top / 2, y_top),
                    (-width_top / 2, y_top)
                ], facecolor=COLORS_MAP[color], edgecolor='none', alpha=0.8)
                ax.add_patch(section)

                # Add percentage label if significant
                if proportions[color] > 0.05:
                    mid_y = y_current + section_height / 2
                    ax.text(0, mid_y, f'{percentages[color]}%',
                            ha='center', va='center', fontsize=10, fontweight='bold')

                y_current = y_top

    # Calculate true proportions
    true_props = {}
    total = len(bag)
    unique_colors = list(set(bag))

    for color in unique_colors:
        true_props[color] = bag.count(color) / total

    # Create figure with two rows
    fig, axes = plt.subplots(2, len(sample_sizes) + 1, figsize=(15, 10))

    # TRUE distribution in both rows
    draw_cup_with_skittles(axes[0, 0], true_props, 60, f"TRUE DISTRIBUTION\n({len(bag)} Skittles)")
    draw_cup_as_stacked_bars(axes[1, 0], true_props)

    # Add vertical separator line after the true distribution
    # Calculate position between first and second column
    separator_x = 0.18  # Adjust this value to fine-tune position
    fig.add_artist(plt.Line2D([separator_x, separator_x], [0.05, 0.95],
                              color='gray', linewidth=2, linestyle='--',
                              transform=fig.transFigure))

    # Show empirical distributions
    for idx, n_samples in enumerate(sample_sizes):
        # Sample n times
        results = {c: 0 for c in unique_colors}
        for _ in range(n_samples):
            results[random.choice(bag)] += 1

        # Convert to proportions
        empirical_props = {color: count / n_samples for color, count in results.items()}

        # Draw both representations
        draw_cup_with_skittles(axes[0, idx + 1], empirical_props, 60, f"After {n_samples} samples")
        draw_cup_as_stacked_bars(axes[1, idx + 1], empirical_props)

    plt.tight_layout()
    plt.show()
