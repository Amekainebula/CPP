#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define double long double
#define ull unsigned long long
#define endl '\n'
using namespace std;
int n, k;
void solve()
{
    cin >> n >> k;
    int mul = n + 1, sum = 0, cur = 1;
    while (n >= k) 
    {
        if (n & 1) sum += cur;
        n >>= 1;
        cur <<= 1;
    }
    cout << mul * sum / 2 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int t;
    cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}