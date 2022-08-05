def main():
    from src.rabbitt_utilities import pwPhase, read_command_line

    args = read_command_line()
    pwPhase(args)


if __name__ == '__main__':
    main()
