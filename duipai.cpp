#include <bits/stdc++.h>
using namespace std;
int main()
{
    system("g++ -std=c++23 -Wl,--stack,536870912 generator.cpp -o generator.exe");
    system("g++ -std=c++23 -Wl,--stack,536870912 brute.cpp -o brute.exe");
    system("g++ -std=c++23 -Wl,--stack,536870912 std.cpp -o std.exe");
    // system("g++ -std=c++23 -Wl,--stack,536870912 check.cpp -o check.exe");
    while (1)
    {

        system("generator.exe > data.txt");
        system("std.exe < data.txt > std.txt");
        // system("check.exe < std.txt > check.txt");
        system("brute.exe < data.txt > brute.txt");
        if (system("fc brute.txt std.txt"))
        {
            cout << "Wrong Answer!" << endl;
            //system("type data.txt");

            system("pause");
            continue;
        }
        cout << "AC!" << endl;
         system("pause");
    }
}
