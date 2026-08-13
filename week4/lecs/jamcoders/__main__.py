"""Command-line entry point: python -m jamcoders [clear_cache|build_cache|list_cache]"""
import sys

from . import build_cache, clear_cache, list_cache

COMMANDS = ('clear_cache', 'build_cache', 'list_cache')


def main(argv):
    if len(argv) >= 2 and argv[1] in ('-h', '--help', 'help'):
        print(f"Usage: python -m jamcoders [{'|'.join(COMMANDS)}]")
        return 0

    if len(argv) < 2 or argv[1] not in COMMANDS:
        if len(argv) >= 2:
            print(f"Unknown command: {argv[1]}")
        print(f"Usage: python -m jamcoders [{'|'.join(COMMANDS)}]")
        return 1

    command = argv[1]
    if command == 'clear_cache':
        clear_cache()
    elif command == 'build_cache':
        return 0 if build_cache() else 1
    else:
        files = list_cache()
        if files:
            print("Cache files:")
            for f in sorted(files):
                print(f"  - {f}")
        else:
            print("No cache files found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
