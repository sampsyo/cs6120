/**
 * A demonstration of a sequential consistency violation. This program behaves
 * one way on a (hypothetical) SC machine and another way on even slightly
 * weaker memory models like TSO.
 *
 * Based on John Regehr's "Simplest Program Showing the Difference Between
 * Sequential Consistency and TSO":
 * http://blog.regehr.org/archives/898
 */

#include <pthread.h>
#include <stdio.h>

volatile int x, y, tmp1, tmp2;

// Thread 0: write x and read y.
void *t0(void *arg) {
  x = 1;
  tmp1 = y;
  return 0;
}

// Thread 1, the opposite: write y and read x.
void *t1(void *arg) {
  y = 1;
  tmp2 = x;
  return 0;
}

int main() {
  while (1) {
    // Initialize all four variables to zero.
    x = y = tmp1 = tmp2 = 0;

    // Launch both threads.
    pthread_t thread0, thread1;
    pthread_create(&thread0, NULL, t0, NULL);
    pthread_create(&thread1, NULL, t1, NULL);

    // Wait for both threads to finish.
    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);

    // Show the results. In any sequentially consistent execution, at least
    // one of the privatized copies (tmp1 and tmp2) will be nonzero. If they
    // are *both* zero, we have violated SC.
    printf("%d %d\n", tmp1, tmp2);
    if (tmp1==0 && tmp2==0) break;
  }
  return 0;
}
