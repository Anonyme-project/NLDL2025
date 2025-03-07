from Helpers.utils import Utils

# =============================================================================
# Main Script
# =============================================================================
if __name__ == "__main__":

    u = Utils()

    output_dir = u.mkdir(["Outputs", "Current"], cwd=True, reset=True)
    print(f"{output_dir=}")
