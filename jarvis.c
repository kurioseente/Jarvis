#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W1_ROWS 784
#define W1_COLS 10
#define W2_ROWS 10
#define W2_COLS 10
#define B1_SIZE 10
#define B2_SIZE 10

double W1[W1_ROWS][W1_COLS];
double W2[W2_ROWS][W2_COLS];
double b1[B1_SIZE];
double b2[B2_SIZE];

double z1[10];
double z2[10];
double a1[10];
double a2[10];

int current_training = 0;

int first_time_init = 1;

void load_params()
{
    if (first_time_init)
    {
        for (int i = 0; i < B1_SIZE; i++)
        {
            b1[i] = 0.0;
        }
        for (int i = 0; i < B2_SIZE; i++)
        {
            b2[i] = 0.0;
        }

        // Weights zufällig initialisieren
        for (int i = 0; i < W1_ROWS; i++)
        {
            for (int j = 0; j < W1_COLS; j++)
            {
                W1[i][j] = -0.087 + ((double)rand() / RAND_MAX) * (0.087 - (-0.087));
            }
        }
        for (int i = 0; i < W2_ROWS; i++)
        {
            for (int j = 0; j < W2_COLS; j++)
            {
                W2[i][j] = -0.548 + ((double)rand() / RAND_MAX) * (0.548 - (-0.548));
            }
        }
    }

    // weights1
    FILE *weights1_ptr = fopen("weights1.csv", "r");
    if (weights1_ptr == NULL)
    {
        printf("Error opening weights1.csv!\n");
        return;
    }
    for (int i = 0; i < W1_ROWS; i++)
    {
        for (int j = 0; j < W1_COLS; j++)
        {
            fscanf(weights1_ptr, "%lf,", &W1[i][j]);
        }
    }
    fclose(weights1_ptr);

    // weights2
    FILE *weights2_ptr = fopen("weights2.csv", "r");
    if (weights2_ptr == NULL)
    {
        printf("Error opening weights2.csv!\n");
        return;
    }
    for (int i = 0; i < W2_ROWS; i++)
    {
        for (int j = 0; j < W2_COLS; j++)
        {
            fscanf(weights2_ptr, "%lf,", &W2[i][j]);
        }
    }
    fclose(weights2_ptr);

    // bias1
    FILE *bias1_ptr = fopen("bias1.csv", "r");
    if (bias1_ptr == NULL)
    {
        printf("Error opening bias1.csv!\n");
        return;
    }
    for (int i = 0; i < B1_SIZE; i++)
    {
        fscanf(bias1_ptr, "%lf,", &b1[i]);
    }
    fclose(bias1_ptr);

    // bias2
    FILE *bias2_ptr = fopen("bias2.csv", "r");
    if (bias2_ptr == NULL)
    {
        printf("Error opening bias2.csv!\n");
        return;
    }
    for (int i = 0; i < B2_SIZE; i++)
    {
        fscanf(bias2_ptr, "%lf,", &b2[i]);
    }
    fclose(bias2_ptr);
}

void save_params()
{

    // weights1
    FILE *weights1_ptr;
    weights1_ptr = fopen("weights1.csv", "w+");
    if (weights1_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < W1_ROWS; i++)
    {
        for (int j = 0; j < W1_COLS; j++)
        {
            fprintf(weights1_ptr, "%.6lf", W1[i][j]);
            if (j < W1_COLS - 1)
                fprintf(weights1_ptr, ",");
        }
        fprintf(weights1_ptr, "\n");
    }
    fclose(weights1_ptr);
    // weights2
    FILE *weights2_ptr;
    weights2_ptr = fopen("weights2.csv", "w+");
    if (weights2_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < W2_ROWS; i++)
    {
        for (int j = 0; j < W2_COLS; j++)
        {
            fprintf(weights2_ptr, "%.6lf", W2[i][j]);
            if (j < W2_COLS - 1)
                fprintf(weights2_ptr, ",");
        }
        fprintf(weights2_ptr, "\n");
    }
    fclose(weights2_ptr);
    // bias1
    FILE *bias1_ptr;
    bias1_ptr = fopen("bias1.csv", "w+");
    if (bias1_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }
    for (int i = 0; i < B1_SIZE; i++)
    {
        fprintf(bias1_ptr, "%.6lf", b1[i]);
        fprintf(bias1_ptr, ",");
    }
    // bias2
    FILE *bias2_ptr;
    bias2_ptr = fopen("bias2.csv", "w+");
    if (bias2_ptr == NULL)
    {
        printf("Error opening file!\n");
        return;
    }
    for (int i = 0; i < B2_SIZE; i++)
    {
        fprintf(bias2_ptr, "%.6lf", b2[i]);
        fprintf(bias2_ptr, ",");
    }
}

// Math
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0;
        for (int j = 0; j < cols; j++)
        {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}
void matrix_add_vector(double *matrix, double *vector, double *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {

        for (int j = 0; j < cols; j++)
        {
            result[i * cols + j] = matrix[i * cols + j] + vector[i];
        }
    }
}
//

double *softmax(double *array, int size)
{
    double *result = (double *)malloc(size * sizeof(double));
    if (result == NULL)
    {
        printf("NUNUNUNUJU fehlgeschlagen\n");
        return NULL;
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        sum += exp(array[i]);
    }

    for (int i = 0; i < size; i++)
    {
        result[i] = exp(array[i]) / sum;
    }

    return result;
}

double ReLu(double n)
{
    return n > 0 ? n : 0;
}


void forward_propagation(int *pic)
{
    // hidden layer
    for (size_t i = 0; i < 10; i++)
    {
        z1[i] = 0.0;
        for (int j = 0; j < 784; j++)
        {
            z1[i] += W1[i][j] * pic[j];
        }
        z1[i] += b1[i];
    }
    for (size_t i = 0; i < 10; i++)
    {
        a1[i] = ReLu(z1[i]);
    }
    for (size_t i = 0; i < 10; i++)
    {
        z2[i] = 0.0;
        for (int j = 0; j < 10; j++)
        {
            z2[i] += W2[i][j] * a1[j];
        }
        z2[i] += b2[i];
    }
    double *softmax_output = softmax(z2, 10);

    for (int i = 0; i < 10; i++)
    {
        a2[i] = softmax_output[i];
    }
    return;
    // a2 waere das durch den softmaz gehauen
}

// wenn der richtige 5 waere saehe das so aus {0,0,0,0,1,0,0,0,0,0}
double fault_rate(double *a2, double *right_one)
{
    double L = 0.0;
    for (int i = 0; i < 10; i++)
    {
        L -= right_one[i] * log(a2[i]);
    }
    return L;
}

//*** where the real struggle & magic happens ngl ***//
void back_propagation(double *right_one, int *pic)
{
    double delta2[10];
    for (int i = 0; i < 10; i++)
    {
        delta2[i] = a2[i] - right_one[i];
    }

    // schleifstein für W2
    double dW2[10][10];
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            dW2[i][j] = delta2[i] * a1[j];
        }
    }

    // schleifstein für b2
    double db2[10];
    for (int i = 0; i < 10; i++)
    {
        db2[i] = delta2[i];
    }

    // Fehler in der Hidde layer
    double delta1[10];
    for (int i = 0; i < 10; i++)
    {
        delta1[i] = 0.0;
        for (int j = 0; j < 10; j++)
        {
            delta1[i] += W2[j][i] * delta2[j];
        }
        delta1[i] *= (z1[i] > 0 ? 1 : 0);
    }

    // Gradienten für W1
    double dW1[10][784];
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 784; j++)
        {
            dW1[i][j] = delta1[i] * pic[j];
        }
    }

    // Gradienten für b1
    double db1[10];
    for (int i = 0; i < 10; i++)
    {
        db1[i] = delta1[i];

        // Lernrate
        double learning_rate = 0.01;

        // Hier wird gelernt also die Gewichte und bias werden angepasst
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 784; j++)
            {
                W1[i][j] -= learning_rate * dW1[i][j];
            }
            b1[i] -= learning_rate * db1[i];
        }

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                W2[i][j] -= learning_rate * dW2[i][j];
            }
            b2[i] -= learning_rate * db2[i];
        }
    }
}

void load_data(int **data_set)
{
    FILE *file = fopen("data.csv", "r");
    if (!file)
    {
        printf("FEHLER: Datei nicht gefunden!\n");
        perror("Fehler");
        return;
    }

    char line[5000];
    int row = 0;

    while (fgets(line, sizeof(line), file) && row < W1_ROWS + 1)
    {
        char *token = strtok(line, ",\r\n");
        int col = 0;

        while (token != NULL && col < W1_COLS)
        {
            data_set[row][col] = atoi(token);
            token = strtok(NULL, ",\r\n");
            col++;
        }
        row++;
    }
    if (row < 5)
    {
        printf("FEHLER: Keine Daten geladen!\n");
    }

    load_params();

    fclose(file);
}



int main()
{
    printf("Good morning, sir. All systems operational.\n");

    // WICHTIG am ende ist es so das bei data[Pixel + Expected answer (1)][Bild] heisst das max in der zukunft das nicht vergisst/vertauscht und 10000 stunden debuggt!!!!!!
    int **data = malloc((W1_ROWS) * sizeof(int *));
    for (int i = 0; i < (W1_ROWS); i++)
    {
        data[i] = malloc(W1_COLS * sizeof(int));
    }

    load_data(data);

    for (int i = 0; i < (W1_ROWS + 1); i++)
    {
        free(data[i]);
    }
    free(data);
    // the number of training exmaples you wanna go through
    for (int i = 0; i < 1000; i++)
    {
        current_training = i;
        int pic[783];
        for (int j = 0; j < 783; j++)
        {
            pic[j] = data[j + 1][current_training];
        }

        forward_propagation(pic);
        double right_one[10];
        for (int i = 0; i < 10; i++)
        {
            if (data[0][current_training] = i)
            {
                right_one[i] = 1;
                continue;
            }
            right_one[i] = 0;
            
        }
        
        back_propagation(right_one, pic);
    }
    printf("hier sind wir");
    save_params();

    return 0;
}