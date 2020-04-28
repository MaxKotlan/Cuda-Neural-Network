//*
//* To enable debug printing uncomment the line below:
//#define DEBUG_ENABLE
//*


#ifdef DEBUG_ENABLE
#define DEBUG(x) do { std::cerr << x; } while (0)
#else
#define DEBUG(x) {}
#endif