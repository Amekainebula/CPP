#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
void Murasame()
{
    int n, m;
    cin >> n >> m;
    vc<vi> ans(n + 1, vi(m + 1));
    // int k = 10;
    // cout << k << " ";
    // ff(i, 1, n)
    // {
    //     k ^= 1;
    //     cout << k << " ";
    // }
    ff(i, 1, n - 1) ff(j, 1, m - 1) ans[i][j] = 1;
    if (n > 2 && m > 2)
    {
        ff(i, 1, n)
        {
            ans[i][m] = i;
            if ((m - 1) % 2)
                ans[i][m] ^= 1;
        }
        ff(j, 1, m)
        {
            ans[n][j] = j + n;
            if ((n - 1) % 2);
        }
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}