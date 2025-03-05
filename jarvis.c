#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W1_ROWS 784
#define W1_COLS 10
#define W2_ROWS 10
#define W2_COLS 10
#define B1_SIZE 10
#define B2_SIZE 10

int W1[W1_ROWS][W1_COLS];
int W2[W2_ROWS][W2_COLS];
int b1[B1_SIZE];
int b2[B2_SIZE];

void load_params(){
    /* wenn ich irgendwann params habe :c */
}


void save_params(){
    FILE *weights1_ptr;
    weights1_ptr = fopen("weights1.csv", "w+");
    if (weights1_ptr == NULL) {
        printf("Error opening file!\n");
        return;
    }

    for(int i = 0; i < W1_ROWS; i++) {
        for(int j = 0; j < W1_COLS; j++) {
            fprintf(weights1_ptr , "%d", W1[i][j]);
            if (j < W1_COLS - 1) fprintf(weights1_ptr, ",");
        }
        fprintf(weights1_ptr, "\n");
    }
    fclose(weights1_ptr);
}

int *softmax(){
    // TODO
    // init hier den array und geb den pointer weiter :thumbs_up:
    return 0;
}

int ReLu(int n){
    return max(0 , n);
}


void load_data(int **array) {
    FILE *file = fopen("data.csv", "r");
    if (!file) {
        printf("FEHLER: Datei nicht gefunden!\n");
        perror("Fehler");
        return;
    }

    char line[5000];
    int row = 0;
   
    while (fgets(line, sizeof(line), file) && row < W1_ROWS + 1) {
        char *token = strtok(line, ",\r\n");
        int col = 0;

        while (token != NULL && col < W1_COLS) {
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

    // WICHTIG am ende ist es so das bei data[Pixel + 1][Bild] heisst das max in der zukunft das nicht vergisst/vertauscht und 10000 stunden debuggt!!!!!!
    int **data = malloc((W1_ROWS+1) * sizeof(int *));
    for (int i = 0; i < (W1_ROWS+1); i++) {
        data[i] = malloc(W1_COLS * sizeof(int));
    }

    load_data(data);
    printf("Wert: %d\n", data[0][0]);

    for (int i = 0; i < (W1_ROWS+1); i++) {
        free(data[i]);
    }
    free(data);

    return 0;
}