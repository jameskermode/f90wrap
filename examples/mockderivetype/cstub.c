// f90wrap error_abort() handler

#include <setjmp.h>
jmp_buf environment_buffer;
char abort_message[1024];

void f90wrap_error_abort_(char *message, int len)
{
  strncpy(abort_message, message, len);
  abort_message[len] = '\0';
  longjmp(environment_buffer, 0);
}

void f90wrap_error_abort_int_handler(int signum)
{
  char message[] = "Interrupt occured";
  f90wrap_error_abort_(message, strlen(message));
}

