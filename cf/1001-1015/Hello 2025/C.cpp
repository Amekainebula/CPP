#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define ll long long
#define ld long double
#define ull unsigned long long
#define endl '\n'
using namespace std;

void solve()
{
    int l, r;
    cin >> l >> r;
    int k = 31 - __builtin_clz(l ^ r);
    int a = l | ((1 << k) - 1), b = a + 1, c = (a == l ? r : l);
    std::cout << a << " " << b << " " << c << endl;
    /*int a, b, c;
    cin >> a >> b >> c;
    int a1 = a ^ b;
    int a2 = a ^ c;
    int b1 = b ^ c;
    cout << a1 << " " << a2 << " " << b1 << endl;
    cout << a1 + a2 + b1 << endl;*/
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    ll T;
    cin >> T;
    //T = 1;
    while (T--)
    {
        solve();
    }
    return 0;
}
//965321865 375544086 12551794 1073741823
//111001100010011010010010001001
//010110011000100101100100010110
//000000101111111000011001110010
//111111111111111111111111111111
//965321865 375544086 12551794
//803995039 959849211 383639396
//2147483646
//1073741823 0 536870911
//1073741823 536870912 536870911
//2147483646
