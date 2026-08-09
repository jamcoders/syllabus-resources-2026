import pytest
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from jamcoders.random import visualize  # Assuming the main file is named skittles_visualization.py


class TestSkittlesVisualization:
    """Test suite for the Skittles visualization function."""

    def teardown_method(self):
        """Close all matplotlib figures after each test."""
        plt.close('all')

    def test_tiny_bag(self):
        """Test visualization with extremely small bag (4 Skittles)."""
        tiny_bag = ["red", "red", "blue", "green"]
        sample_sizes = [2, 4, 10, 20]

        # Should not raise any exceptions
        visualize(tiny_bag, sample_sizes)

        # Check that a figure was created
        assert len(plt.get_fignums()) > 0

    def test_small_bag(self):
        """Test visualization with small bag (10 Skittles)."""
        small_bag = ["red"] * 4 + ["blue"] * 3 + ["green"] * 2 + ["yellow"] * 1
        sample_sizes = [5, 10, 50, 100]

        visualize(small_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_medium_bag(self):
        """Test visualization with medium bag (100 Skittles) - original size."""
        medium_bag = ["red"] * 40 + ["blue"] * 30 + ["green"] * 20 + ["yellow"] * 10
        sample_sizes = [50, 200, 1000, 5000]

        visualize(medium_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_large_bag(self):
        """Test visualization with large bag (1000 Skittles)."""
        large_bag = ["red"] * 400 + ["blue"] * 300 + ["green"] * 200 + ["yellow"] * 100
        sample_sizes = [100, 500, 2000, 10000]

        visualize(large_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    @pytest.mark.slow
    def test_huge_bag(self):
        """Test visualization with huge bag (10,000 Skittles)."""
        huge_bag = ["red"] * 4000 + ["blue"] * 3000 + ["green"] * 2000 + ["yellow"] * 1000
        sample_sizes = [500, 2000, 5000, 20000]

        visualize(huge_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_single_skittle(self):
        """Test edge case with single Skittle."""
        single_bag = ["red"]
        sample_sizes = [1, 1, 1, 1]

        visualize(single_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_two_colors_only(self):
        """Test visualization with only two colors."""
        two_color_bag = ["red"] * 30 + ["blue"] * 20
        sample_sizes = [10, 25, 100, 500]

        visualize(two_color_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_oversampling(self):
        """Test sampling more times than Skittles in bag."""
        small_bag = ["red"] * 5 + ["blue"] * 5
        sample_sizes = [5, 10, 50, 100]  # 50 and 100 are larger than bag size

        visualize(small_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_empty_bag(self):
        """Test edge case with empty bag."""
        empty_bag = []
        sample_sizes = [10, 20, 30, 40]

        with pytest.raises(IndexError):
            visualize(empty_bag, sample_sizes)

    def test_single_color_bag(self):
        """Test bag with only one color."""
        single_color_bag = ["red"] * 50
        sample_sizes = [10, 25, 100, 200]

        visualize(single_color_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_uneven_distribution(self):
        """Test heavily skewed distribution."""
        skewed_bag = ["red"] * 95 + ["blue"] * 3 + ["green"] * 1 + ["yellow"] * 1
        sample_sizes = [50, 100, 500, 1000]

        visualize(skewed_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_missing_color(self):
        """Test bag missing one of the standard colors."""
        three_color_bag = ["red"] * 33 + ["blue"] * 33 + ["green"] * 34
        # No yellow
        sample_sizes = [50, 100, 500, 1000]

        visualize(three_color_bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    @pytest.mark.parametrize("bag_size,sample_sizes", [
        (20, [5, 10, 50, 100]),
        (50, [10, 25, 100, 500]),
        (200, [50, 100, 500, 2000]),
        (500, [100, 250, 1000, 5000]),
    ])
    def test_various_bag_sizes(self, bag_size, sample_sizes):
        """Parametrized test for various bag sizes."""
        # Create bag with proportional distribution
        bag = []
        bag += ["red"] * int(bag_size * 0.4)
        bag += ["blue"] * int(bag_size * 0.3)
        bag += ["green"] * int(bag_size * 0.2)
        bag += ["yellow"] * int(bag_size * 0.1)

        visualize(bag, sample_sizes)
        assert len(plt.get_fignums()) > 0

    def test_invalid_sample_sizes(self):
        """Test with invalid sample sizes."""
        bag = ["red"] * 50 + ["blue"] * 50

        # Negative sample sizes should raise ValueError when sampling
        with pytest.raises(ValueError):
            sample_sizes = [-10, 20, 30, 40]
            visualize(bag, sample_sizes)

    def test_figure_properties(self):
        """Test that the figure has expected properties."""
        bag = ["red"] * 40 + ["blue"] * 30 + ["green"] * 20 + ["yellow"] * 10
        sample_sizes = [50, 200, 1000, 5000]

        visualize(bag, sample_sizes)

        fig = plt.gcf()
        # Should have 2 rows and len(sample_sizes) + 1 columns
        assert fig.get_axes().__len__() == 2 * (len(sample_sizes) + 1)

        # Check figure size
        assert fig.get_size_inches()[0] == 15  # width
        assert fig.get_size_inches()[1] == 10  # height