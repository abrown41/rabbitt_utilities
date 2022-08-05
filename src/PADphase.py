def main():
    from src.rabbitt_utilities import PADphase, read_command_line
    args = read_command_line()
    PADphase(args)


if __name__ == '__main__':
    main()
