from get_database import format_data
from functions import RAG, get_response


def main():

    #format_data()

    while True:

        query = input("Enter your query: ").strip()

        top_chunks = RAG(query, "../Database/chunk.pkl", n_chunks=5)

        response, context = get_response(query, top_chunks, "Energy Transition")

        print("Response: ", response)

        show_context = input("Would you like to take a look at the paper's summary? (y/n)").strip()

        if show_context == "y":
            print("Context: ", context)

        print(context)

        while True:
            # Ask the user if they want to continue
            more_info = input("Would you like to ask more questions? (y/n) ").strip().lower()

            if more_info == "y":
                # If the user wants to ask more questions, break to the outer loop
                break
            elif more_info == "n":
                # If the user doesn't want more questions, exit the program
                print("Goodbye!")
                exit()  # Exiting the loop and ending the program
            else:
                # Prompt again if input is invalid
                print("Please answer only with 'y' or 'n'")


if __name__ == "__main__":
    main()
