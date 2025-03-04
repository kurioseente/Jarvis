#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 50000
#define COLS 784

void load_data(int **array) { 
    FILE *file = fopen("data.csv", "r");
    if (!file) {
        printf("FEHLER: Datei nicht gefunden!\n");
        perror("Fehler");
        return;
    }

    char line[5000]; 
    int row = 0;
    
    while (fgets(line, sizeof(line), file) && row < ROWS) {
        char *token = strtok(line, ",\r\n");
        int col = 0;

        while (token != NULL && col < COLS) {
            array[row][col] = atoi(token);
            token = strtok(NULL, ",\r\n");
            col++;
        }
        row++;
    }

    if (row < 5) {
        printf("FEHLER: Keine Daten geladen!\n");
    }

    fclose(file);
}

int main() {
    printf("Good morning, sir. All systems operational.\n");

    // WICHTIG am ende ist es so das bei data[Pixel][Bild] heisst das max in der zukunft das nicht vergisst/vertauscht und 10000 stunden debuggt!!!!!!
    int **data = malloc(ROWS * sizeof(int *));
    for (int i = 0; i < ROWS; i++) {
        data[i] = malloc(COLS * sizeof(int));
    }

    load_data(data);
    printf("Wert: %d\n", data[0][0]);

    for (int i = 0; i < ROWS; i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}
