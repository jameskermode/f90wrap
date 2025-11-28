/* custom abort handler - James Kermode <james.kermode@gmail.com> */

#ifdef __cplusplus
extern "C"
{
#endif

#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#define ABORT_BUFFER_SIZE 1024
extern jmp_buf environment_buffer;
extern char abort_message[ABORT_BUFFER_SIZE];
jmp_buf environment_buffer;
char abort_message[ABORT_BUFFER_SIZE];

#ifdef __cplusplus
}
#endif

/* end of custom abort handler  */
