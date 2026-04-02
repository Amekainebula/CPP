#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define double long double
#define ull unsigned long long
#define endl '\n'
using namespace std;
int mod = 998244353;
int t;
int nn(int n) 
{
    int result = 1;
    for (int i = 2; i <= n; i++) 
    {
        result = (result * i) % mod;
    }
    return result;
}

int ans[100];
signed main()
{
    cin >> t;
    while (t--)
    {
        memset(ans, 0, sizeof(ans));
        int n, d;
        cin >> n >> d;
        for (int i = 1; i <= 9; i++)
            if (d == i)ans[i] = 1;
        ans[1] = 1;
        if (n >= 3)
            ans[3] = 1; 
        if (n >= 3 && (d == 3 || d == 6) || n >= 6)
            ans[9] = 1;
        if (n >= 3)
            ans[7] = 1;
        if (n <= 3)
        {
            int tt = nn(n);
            int dd = 0;
            for (int i = 1; i <= tt; i++)
            {
                dd = dd * 10 + d;
            }
            for (int i = 1; i <= 9; i++)
            {
                if (dd % i == 0)ans[i] = 1;
                else ans[i] = 0;
            }
        }
        for (int i = 1; i <= 9; i++)
        {
            if (ans[i] == 1 && i % 2 == 1)cout << i << " ";
        }
        cout << endl;
    }
    return 0;
}
