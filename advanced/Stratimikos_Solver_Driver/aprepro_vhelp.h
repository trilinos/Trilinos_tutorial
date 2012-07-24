/**/ //additional code to solve undefined lib problem 
extern "C" void version(char *vstring)
{
}
/* Global options */
struct aprepro_options
{
   char comment;
   char *include_path;

   int end_on_exit;
   int warning_msg;
   int info_msg;
   int copyright;
   int quiet;
   int debugging;
   int statistics;
   int interactive;
};

typedef struct aprepro_options aprepro_options;

aprepro_options ap_options;
/**/
