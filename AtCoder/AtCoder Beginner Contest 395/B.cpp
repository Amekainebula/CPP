#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
char maps[55][55];
void solve()
{
    int n;
    cin >> n;
    int cnt = 0;
    while (cnt <= n / 2)
    {
        if (cnt % 2 == 0)
        {
            for (int i = 1 + cnt; i <= n - cnt; i++)
            {
                if (i == 1 + cnt || i == n - cnt)
                {
                    for (int j = 1 + cnt; j <= n - cnt; j++)
                        maps[i][j] = '#';
                }
                else
                {
                    maps[i][1 + cnt] = '#';
                    maps[i][n - cnt] = '#';
                }
            }
        }
        else
        {
            for (int i = 1 + cnt; i <= n - cnt; i++)
            {
                if (i == 1 + cnt || i == n - cnt)
                {
                    for (int j = 1 + cnt; j <= n - cnt; j++)
                        maps[i][j] = '.';
                }
                else
                {
                    maps[i][1 + cnt] = '.';
                    maps[i][n - cnt] = '.';
                }
            }
        }
        cnt++;
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
            cout << maps[i][j];
        cout << endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}