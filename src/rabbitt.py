def main():
    from src.rabbitt_utilities import rabbitt, read_command_line

    args = read_command_line()
    rabbitt(args)


if __name__ == '__main__':
    main()
