NAME          PROBLEM
ROWS
 N  OBJ
 L  C1
 G  C2
 L  C3
 G  C4
 L  C5
 L  C6
 L  C7
COLUMNS
 Y1     OBJ      0
 Y1     C1       1
 Y1     C2      -1
 Y1     C5       1
 Y1     C6      -1
 Y2     OBJ      1
 Y2     C3       1
 Y2     C4      -1
 Y2     C5     -1.0
 Y2     C6      1.0
RHS
 RHS1   C1      -1
 RHS1   C2       1
 RHS1   C3      0.999999
 RHS1   C4      -1
 RHS1   C5       1
 RHS1   C6       1
BOUNDS
 LO BND Y1      -1
 UP BND Y1       1
 LO BND Y2      -1
 UP BND Y2      0.999999
ENDATA