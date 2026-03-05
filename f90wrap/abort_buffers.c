/* custom abort handler - James Kermode <james.kermode@gmail.com> */

#ifdef __cplusplus
extern "C"
{
#endif

#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#define ABORT_BUFFER_SIZE 1024
jmp_buf environment_buffer;
char abort_message[ABORT_BUFFER_SIZE];

void f90wrap_abort_(char *message, int len_message)
{
  int copy_len = len_message < ABORT_BUFFER_SIZE - 1 ? len_message : ABORT_BUFFER_SIZE - 1;
  strncpy(abort_message, message, copy_len);
  abort_message[copy_len] = '\0';
  longjmp(environment_buffer, 0);
}

// copy of f90wrap_abort_ with a second underscore
void f90wrap_abort__(char *message, int len_message)
{
  int copy_len = len_message < ABORT_BUFFER_SIZE - 1 ? len_message : ABORT_BUFFER_SIZE - 1;
  strncpy(abort_message, message, copy_len);
  abort_message[copy_len] = '\0';
  longjmp(environment_buffer, 0);
}

void f90wrap_abort_int_handler(int signum)
{
  char message[] = "Interrupt occured";
  f90wrap_abort_(message, strlen(message));
}

#ifdef __cplusplus
}
#endif

/* end of custom abort handler  */
