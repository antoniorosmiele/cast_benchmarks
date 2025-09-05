/* Original code taken from https://github.com/impedimentToProgress/MiBench2 and restructured */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stddef.h>

///////////////////////////
//APPROSIMATED DATA ARRAYS
///////////////////////////#define INIMAGE 0
#define GREYIMAGE 1
#define SMOOTHIMAGE 2
#define RIMAGE 3


#define SEVEN_SUPP // Size for non-max corner suppression; SEVEN_SUPP or FIVE_SUPP
#define MAX_CORNERS 1500 // Max corners per frame

#define exit_error(fmt, ...) { printf(fmt, __VA_ARGS__); exit(EXIT_FAILURE); }
#define FTOI(a) ((a) < 0 ? ((int32_t)(a - 0.5)) : ((int32_t)(a + 0.5)))

#define SMOOTHINGSIZE 3

typedef struct {
  int rows;
  int cols;
  int depth;
  int size;
  unsigned char* header;
  unsigned char* data;
} sImage;



void deleteImage(sImage* image);
void readImage(const char* filename, sImage* image);
void rgb2grey (sImage* originalImage, sImage* greyImage);
void initImage(sImage* image, int rows, int cols, int depth, unsigned char* headar);


/*
 * Deletes the memory object
 */
void deleteImage(sImage* image){
  if(image->header)
    free(image->header);
  if(image->data)
    free(image->data);
  image->rows = image->cols = image->depth = 0;
  image->header = image->data = NULL;
}

/*
 * Reads a 24bit RGB image from a file and saves it in a memory object (required memory is instantiated here)
 */
void readImage(const char* filename, sImage* image){
  FILE *bmpInput;
  int bits = 0;
  int fileSize = 0;
  int results;

  bmpInput = fopen(filename, "rb");
  if(bmpInput){

    image->header = (unsigned char *)malloc(sizeof(unsigned char)*54L);
    results=fread(image->header, sizeof(char), 54L, bmpInput);
    if(results!=54L)
      printf("Bmp read error\n");
    //else
    //  printf("header read ok!\n");

    memcpy(&fileSize,&image->header[2],4);
    memcpy(&image->cols,&image->header[18],4);
    memcpy(&image->rows,&image->header[22],4);
    memcpy(&bits,&image->header[28],4);
    image->depth=3;
    image->size = image->rows*image->cols*image->depth;


/*    printf("Width: %d\n", image->cols);
    printf("Height: %d\n", image->rows);
    printf("File size: %d\n", fileSize);
    printf("Bits/pixel: %d\n", bits);
  */
    if(bits!=24 || fileSize!=image->rows*image->cols*image->depth+54){
      printf("Wrong image format in %s: accepted only 24 bit without padding! %d %d %d %d %d\n", filename,bits,fileSize, image->rows, image->cols, image->depth+54);
      exit(1);
    }

    image->data = (unsigned char *)malloc(image->rows*image->cols*sizeof(unsigned char)*3);
    fseek(bmpInput, 54L, SEEK_SET);
    results=fread(image->data, sizeof(char), (image->rows*image->cols*image->depth), bmpInput);
    if(results != (image->rows*image->cols*image->depth))
      printf("Bmp read error\n");
    //else
    //  printf("data read ok %d!\n", results);
    //printf("File read\n");

    fclose(bmpInput);
  } else {
    printf("File not found: %s\n",filename);
    exit(1);
  }
}

/*
 * Initializes the memory object
 */
void initImage(sImage* image, int rows, int cols, int depth, unsigned char* header){
  image->rows = rows;
  image->cols = cols;
  image->depth = depth;
  image->size = rows*cols*depth;
  image->data = (unsigned char *)malloc(sizeof(unsigned char)*rows*cols*depth);
  //memset(image->data, 0, sizeof(unsigned char)*rows*cols*depth); //DO NOTE: for debugging purposes we perform a memset to 0. TODO remove it
  if(header){
    image->header = (unsigned char *)malloc(sizeof(unsigned char)*54L);
    memcpy(image->header,header,54L);
  } else
    image->header = NULL;
}

/*
 * Converts an RGB image in grey-scale format and saves it in another object. Memory has to be already instantiated
 */
void rgb2grey (sImage* originalImage, sImage* greyImage){
  int r = 0;
  int c = 0;
  unsigned char  redValue, greenValue, blueValue, grayValue;

  for(r=0; r<originalImage->rows; r++){
    for (c=0; c<originalImage->cols; c++){
      /*-----READ FIRST BYTE TO GET BLUE VALUE-----*/
      blueValue = *(originalImage->data + (r*originalImage->cols + c)*3);
      /*-----READ NEXT BYTE TO GET GREEN VALUE-----*/
      greenValue = *(originalImage->data + (r*originalImage->cols +c)*3+1);
      /*-----READ NEXT BYTE TO GET RED VALUE-----*/
      redValue = *(originalImage->data + (r*originalImage->cols +c)*3+2);
      /*-----USE FORMULA TO CONVERT RGB VALUE TO GRAYSCALE-----*/
      grayValue = (unsigned char) (0.299*redValue + 0.587*greenValue + 0.114*blueValue);
      *(greyImage->data + r*originalImage->cols +c) = grayValue;
    }
  }
}

void conv2D(unsigned char* inputImage, unsigned char* outputImage, int height, int width, int depth, double* filter, int filter_size) {
  int i, j, k, i1, j1;
  int ci=-filter_size/2, cj=-filter_size/2;
  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      for(k = 0; k < depth; k++) {
        double value = 0;
        for(i1 = 0; i1 < filter_size; i1++) {
          for(j1 = 0; j1 < filter_size; j1++) {
            int i_img = i+i1+ci;
            int j_img = j+j1+cj;
            if(i_img < 0)
              i_img = -i_img;
            else if(i_img >=height)
              i_img = 2*height -2 - i_img;            
            if(j_img < 0)
              j_img = -j_img;
            else if(j_img >=width)
              j_img = 2*width -2 - j_img;            
            value += filter[(i1*filter_size)+j1] * inputImage[(((i_img)*width)+(j_img))*depth+k];              
          }
        }
        outputImage[((i*width)+j)*depth+k] = value;
      }
    }
  }
}

double* generateSmoothingFilter(int size) {
  // Calculate a gaussian blur psf.
  double sigma_row = 9.0;
  double sigma_col = 5.0;
  
  double mean_row = 0.0;
  double mean_col = size/2.0;
  double sum = 0.0;
  double temp;
  
  double* psf = (double*) malloc(sizeof(double)*size*size);
  
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      temp = exp(
          -0.5 * (
            pow((i - mean_row) / sigma_row, 2.0) + 
            pow((j - mean_col) / sigma_col, 2.0))) /
        (2* M_PI * sigma_row * sigma_col);
      sum += temp;
      psf[i*size+j] = temp;
    }
  }

  // Normalise the psf.
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      psf[i*size+j] /= sum;
    }
  }
  return psf;
}

typedef struct {
    int32_t x, y, info, dx, dy, I;
} Corner;

typedef struct {
    Corner corners[MAX_CORNERS];
} CornerList;

//static unsigned char *fakeFile;
static unsigned char setbrightness[516];
static unsigned char smoothening[9810];
static unsigned char smoothening2[225];

/*unsigned char fgetc2() {
    unsigned char ret = *fakeFile;
    ++fakeFile;
    return ret;
}*/

/*int32_t getint() {
    int32_t c, i = 0;
    c = fgetc2();
    while (1) {
        if (c == '#') {
            while (c != '\n') c = fgetc2();
        }
        if (c == EOF) exit_error("Image %s not binary PGM.\n", "is");
        if (c >= '0' && c <= '9') break;
        c = fgetc2();
    }

    while (1) {
        i = (i * 10) + (c - '0');
        c = fgetc2();
        if (c == EOF) return i;
        if (c < '0' || c > '9') break;
    }

    return i;
}*/

/*void get_image(unsigned char **in, int32_t *x_size, int32_t *y_size) {
    unsigned char header[2];
    header[0] = fgetc2();
    header[1] = fgetc2();

    if (!(header[0] == 'P' && header[1] == '5')) {
        exit_error("Image does %s have binary PGM header.\n", "not");
    }

    *x_size = getint();
    *y_size = getint();
    *in = (unsigned char *)fakeFile;
}*/

void put_image(const unsigned char *in, int32_t x_size, int32_t y_size, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
        return;
    }

    fprintf(file, "P5\n");          // PGM magic number (binary graymap)
    fprintf(file, "%d %d\n", x_size, y_size); // Image dimensions
    fprintf(file, "255\n");         // Maximum pixel value (8-bit grayscale)

    fwrite(in, sizeof(unsigned char), x_size * y_size, file);
    fclose(file);

//    printf("Image successfully written to %s\n", filename);
}

void int_to_uchar(const int32_t *r, unsigned char *in, size_t size) {
    int32_t max_r = r[0], min_r = r[0];

    for (size_t i = 0; i < size; i++) {
        if (r[i] > max_r) max_r = r[i];
        if (r[i] < min_r) min_r = r[i];
    }

    max_r -= min_r;

    for (size_t i = 0; i < size; i++) {
        in[i] = (unsigned char)(((r[i] - min_r) * 255) / max_r);
    }
}

void setup_brightness_lut(unsigned char **bp, int32_t thresh, int32_t form) {
    *bp = setbrightness + 258;

    for (int32_t k = -256; k < 257; k++) {
        float temp = ((float)k) / ((float)thresh);
        temp = temp * temp;
        if (form == 6) temp = temp * temp * temp;
        temp = 100.0 * exp(-temp);
        *(*bp + k) = (unsigned char)temp;
    }
}

void corner_draw(unsigned char *in, const CornerList *corner_list, int32_t x_size, int32_t drawing_mode) {
    const Corner *corner = corner_list->corners;
    while (corner->info != 7) {
        if (drawing_mode == 0) {
            unsigned char *p = in + (corner->y - 1) * x_size + corner->x - 1;
            *p++ = 255; *p++ = 255; *p = 255; p += x_size - 2;
            *p++ = 255; *p++ = 0; *p = 255; p += x_size - 2;
            *p++ = 255; *p++ = 255; *p = 255;
        } else {
            *(in + corner->y * x_size + corner->x) = 0;
        }
        corner++;
    }
}

void susan_corners(const unsigned char *in, int32_t *r, const unsigned char *bp, int32_t max_no, CornerList *corner_list, int32_t x_size, int32_t y_size) {
    memset(r, 0, x_size * y_size * sizeof(int32_t));

    int32_t *cgx = (int32_t *)malloc(x_size * y_size * sizeof(int32_t));
    int32_t *cgy = (int32_t *)malloc(x_size * y_size * sizeof(int32_t));
    if (!cgx || !cgy) exit_error("Failed to allocate memory for cgx or cgy.\n", "");

    for (int32_t i = 5; i < y_size - 5; i++) {
        for (int32_t j = 5; j < x_size - 5; j++) {
            int32_t n = 100;
            const unsigned char *p = in + (i - 3) * x_size + j - 1;
            const unsigned char *cp = bp + in[i * x_size + j];

            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p);
            p += x_size - 3;

            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p);
            p += x_size - 5;

            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p);
            p += x_size - 6;

            n += *(cp - *p++);
            n += *(cp - *p++);
            n += *(cp - *p);
            if (n < max_no) {
                p += 2;
                n += *(cp - *p++);
                if (n < max_no) {
                    n += *(cp - *p++);
                    if (n < max_no) {
                        n += *(cp - *p);
                        if (n < max_no) {
                            p += x_size - 6;

                            n += *(cp - *p++);
                            if (n < max_no) {
                                n += *(cp - *p++);
                                if (n < max_no) {
                                    n += *(cp - *p++);
                                    if (n < max_no) {
                                        n += *(cp - *p++);
                                        if (n < max_no) {
                                            n += *(cp - *p++);
                                            if (n < max_no) {
                                                n += *(cp - *p++);
                                                if (n < max_no) {
                                                    n += *(cp - *p);
                                                    if (n < max_no) {
                                                        p += x_size - 5;

                                                        n += *(cp - *p++);
                                                        if (n < max_no) {
                                                            n += *(cp - *p++);
                                                            if (n < max_no) {
                                                                n += *(cp - *p++);
                                                                if (n < max_no) {
                                                                    n += *(cp - *p++);
                                                                    if (n < max_no) {
                                                                        n += *(cp - *p);
                                                                        if (n < max_no) {
                                                                            p += x_size - 3;

                                                                            n += *(cp - *p++);
                                                                            if (n < max_no) {
                                                                                n += *(cp - *p++);
                                                                                if (n < max_no) {
                                                                                    n += *(cp - *p);

                                                                                    if (n < max_no) {
                                                                                        int32_t x = 0, y = 0;
                                                                                        p = in + (i - 3) * x_size + j - 1;

                                                                                        unsigned char c = *(cp - *p++); x -= c; y -= 3 * c;
                                                                                        c = *(cp - *p++); y -= 3 * c;
                                                                                        c = *(cp - *p); x += c; y -= 3 * c;
                                                                                        p += x_size - 3;

                                                                                        c = *(cp - *p++); x -= 2 * c; y -= 2 * c;
                                                                                        c = *(cp - *p++); x -= c; y -= 2 * c;
                                                                                        c = *(cp - *p++); y -= 2 * c;
                                                                                        c = *(cp - *p++); x += c; y -= 2 * c;
                                                                                        c = *(cp - *p); x += 2 * c; y -= 2 * c;
                                                                                        p += x_size - 5;

                                                                                        c = *(cp - *p++); x -= 3 * c; y -= c;
                                                                                        c = *(cp - *p++); x -= 2 * c; y -= c;
                                                                                        c = *(cp - *p++); x -= c; y -= c;
                                                                                        c = *(cp - *p++); y -= c;
                                                                                        c = *(cp - *p++); x += c; y -= c;
                                                                                        c = *(cp - *p++); x += 2 * c; y -= c;
                                                                                        c = *(cp - *p); x += 3 * c; y -= c;
                                                                                        p += x_size - 6;

                                                                                        c = *(cp - *p++); x -= 3 * c;
                                                                                        c = *(cp - *p++); x -= 2 * c;
                                                                                        c = *(cp - *p); x -= c;
                                                                                        p += 2;
                                                                                        c = *(cp - *p++); x += c;
                                                                                        c = *(cp - *p++); x += 2 * c;
                                                                                        c = *(cp - *p); x += 3 * c;
                                                                                        p += x_size - 6;

                                                                                        c = *(cp - *p++); x -= 3 * c; y += c;
                                                                                        c = *(cp - *p++); x -= 2 * c; y += c;
                                                                                        c = *(cp - *p++); x -= c; y += c;
                                                                                        c = *(cp - *p++); y += c;
                                                                                        c = *(cp - *p++); x += c; y += c;
                                                                                        c = *(cp - *p++); x += 2 * c; y += c;
                                                                                        c = *(cp - *p); x += 3 * c; y += c;
                                                                                        p += x_size - 5;

                                                                                        c = *(cp - *p++); x -= 2 * c; y += 2 * c;
                                                                                        c = *(cp - *p++); x -= c; y += 2 * c;
                                                                                        c = *(cp - *p++); y += 2 * c;
                                                                                        c = *(cp - *p++); x += c; y += 2 * c;
                                                                                        c = *(cp - *p); x += 2 * c; y += 2 * c;
                                                                                        p += x_size - 3;

                                                                                        c = *(cp - *p++); x -= c; y += 3 * c;
                                                                                        c = *(cp - *p++); y += 3 * c;
                                                                                        c = *(cp - *p); x += c; y += 3 * c;

                                                                                        int32_t xx = x * x;
                                                                                        int32_t yy = y * y;
                                                                                        int32_t sq = xx + yy;
                                                                                        if (sq > ((n * n) / 2)) {
                                                                                            if (yy < xx) {
                                                                                                float divide = (float)y / (float)abs(x);
                                                                                                sq = abs(x) / x;
                                                                                                sq = *(cp - in[(i + FTOI(divide)) * x_size + j + sq]) +
                                                                                                     *(cp - in[(i + FTOI(2 * divide)) * x_size + j + 2 * sq]) +
                                                                                                     *(cp - in[(i + FTOI(3 * divide)) * x_size + j + 3 * sq]);
                                                                                            } else {
                                                                                                float divide = (float)x / (float)abs(y);
                                                                                                sq = abs(y) / y;
                                                                                                sq = *(cp - in[(i + sq) * x_size + j + FTOI(divide)]) +
                                                                                                     *(cp - in[(i + 2 * sq) * x_size + j + FTOI(2 * divide)]) +
                                                                                                     *(cp - in[(i + 3 * sq) * x_size + j + FTOI(3 * divide)]);
                                                                                            }

                                                                                            if (sq > 290) {
                                                                                                r[i * x_size + j] = max_no - n;
                                                                                                cgx[i * x_size + j] = (51 * x) / n;
                                                                                                cgy[i * x_size + j] = (51 * y) / n;
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    int32_t n = 0;
    for (int32_t i = 5; i < y_size - 5; i++) {
        for (int32_t j = 5; j < x_size - 5; j++) {
            int32_t x = r[i * x_size + j];
            if (x > 0) {
#ifdef SEVEN_SUPP
                if (
                    (x > r[(i - 3) * x_size + j - 3]) &&
                    (x > r[(i - 3) * x_size + j - 2]) &&
                    (x > r[(i - 3) * x_size + j - 1]) &&
                    (x > r[(i - 3) * x_size + j]) &&
                    (x > r[(i - 3) * x_size + j + 1]) &&
                    (x > r[(i - 3) * x_size + j + 2]) &&
                    (x > r[(i - 3) * x_size + j + 3]) &&

                    (x > r[(i - 2) * x_size + j - 3]) &&
                    (x > r[(i - 2) * x_size + j - 2]) &&
                    (x > r[(i - 2) * x_size + j - 1]) &&
                    (x > r[(i - 2) * x_size + j]) &&
                    (x > r[(i - 2) * x_size + j + 1]) &&
                    (x > r[(i - 2) * x_size + j + 2]) &&
                    (x > r[(i - 2) * x_size + j + 3]) &&

                    (x > r[(i - 1) * x_size + j - 3]) &&
                    (x > r[(i - 1) * x_size + j - 2]) &&
                    (x > r[(i - 1) * x_size + j - 1]) &&
                    (x > r[(i - 1) * x_size + j]) &&
                    (x > r[(i - 1) * x_size + j + 1]) &&
                    (x > r[(i - 1) * x_size + j + 2]) &&
                    (x > r[(i - 1) * x_size + j + 3]) &&

                    (x > r[(i)*x_size + j - 3]) &&
                    (x > r[(i)*x_size + j - 2]) &&
                    (x > r[(i)*x_size + j - 1]) &&
                    (x >= r[(i)*x_size + j + 1]) &&
                    (x >= r[(i)*x_size + j + 2]) &&
                    (x >= r[(i)*x_size + j + 3]) &&

                    (x >= r[(i + 1) * x_size + j - 3]) &&
                    (x >= r[(i + 1) * x_size + j - 2]) &&
                    (x >= r[(i + 1) * x_size + j - 1]) &&
                    (x >= r[(i + 1) * x_size + j]) &&
                    (x >= r[(i + 1) * x_size + j + 1]) &&
                    (x >= r[(i + 1) * x_size + j + 2]) &&
                    (x >= r[(i + 1) * x_size + j + 3]) &&

                    (x >= r[(i + 2) * x_size + j - 3]) &&
                    (x >= r[(i + 2) * x_size + j - 2]) &&
                    (x >= r[(i + 2) * x_size + j - 1]) &&
                    (x >= r[(i + 2) * x_size + j]) &&
                    (x >= r[(i + 2) * x_size + j + 1]) &&
                    (x >= r[(i + 2) * x_size + j + 2]) &&
                    (x >= r[(i + 2) * x_size + j + 3]) &&

                    (x >= r[(i + 3) * x_size + j - 3]) &&
                    (x >= r[(i + 3) * x_size + j - 2]) &&
                    (x >= r[(i + 3) * x_size + j - 1]) &&
                    (x >= r[(i + 3) * x_size + j]) &&
                    (x >= r[(i + 3) * x_size + j + 1]) &&
                    (x >= r[(i + 3) * x_size + j + 2]) &&
                    (x >= r[(i + 3) * x_size + j + 3])) {
                    corner_list->corners[n].info = 0;
                    corner_list->corners[n].x = j;
                    corner_list->corners[n].y = i;
                    corner_list->corners[n].dx = cgx[i * x_size + j];
                    corner_list->corners[n].dy = cgy[i * x_size + j];
                    corner_list->corners[n].I = in[i * x_size + j];
                    n++;
                    if (n == MAX_CORNERS) {
                        printf("Too many corners.\n");
                        exit(1);
                    }
                }
#endif
            }
        }
    }
    corner_list->corners[n].info = 7;

    free(cgx);
    free(cgy);
}

int main(int argc, char **argv) {
    if(argc!=2){
        printf("USAGE: %s input.bmp\n", argv[0]);
        return 1;
    }
    unsigned char *in;
    int32_t x_size = -1, y_size = -1;

//    fakeFile = test_data; // Assuming test_data is defined elsewhere
//    get_image(&in, &x_size, &y_size);

    sImage inImage, greyImage, smoothImage;

    double* filter = generateSmoothingFilter(SMOOTHINGSIZE);


    readImage(argv[1], &inImage);
    initImage(&greyImage, inImage.rows, inImage.cols, 1, NULL);
    initImage(&smoothImage, inImage.rows, inImage.cols, 1, NULL);
    rgb2grey(&inImage,&greyImage);
    conv2D(greyImage.data, smoothImage.data, inImage.rows, inImage.cols, 1, filter, SMOOTHINGSIZE);


    x_size = inImage.cols;
    y_size = inImage.rows;

    in = smoothImage.data;

    int32_t *r = (int32_t *)malloc(x_size * y_size * sizeof(int32_t));
    if (!r) exit_error("Failed to allocate memory for r.\n", "");

    unsigned char *bp;
    setup_brightness_lut(&bp, 20, 6);

    CornerList corner_list;
    susan_corners(in, r, bp, 1850, &corner_list, x_size, y_size);
    //corner_draw(in, &corner_list, x_size, 0);
    //put_image(in, x_size, y_size, "output.pgm");

    FILE* fp;
    fp = fopen("golden_output.txt", "w");
    if(fp){
        for(int i=0; corner_list.corners[i].info!=7; i++)
            fprintf(fp, "%d %d\n", corner_list.corners[i].x, corner_list.corners[i].y);
        fclose(fp);
    }

    deleteImage(&greyImage);
    deleteImage(&smoothImage);
    deleteImage(&inImage);

    free(r);
    free(filter);

    return 0;
}
