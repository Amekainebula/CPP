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
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void Murasame()
{
    int n;
    cin >> n;
    vc<vc<int>> maps(n + 1, vi(n + 1));
    ff(i, 1, n) ff(j, 1, n) cin >> maps[i][j];
    vi a(n + 1), b(n + 1);
    ff(i, 1, n) cin >> a[i];
    ff(i, 1, n) cin >> b[i];
    auto f = [&](vi &val)
    {
        vc<vc<int>> dp(n + 1, vi(2, INF / 2));
        dp[1][0] = 0;
        dp[1][1] = val[1];
        ff(i, 2, n) ff(x, 0, 1) ff(y, 0, 1)
        {
            bool ok = 1;
            ff(j, 1, n)
                ok &= (maps[i][j] + x != maps[i - 1][j] + y);
            if (ok)
            {
                if (x == 0)
                    dp[i][x] = min(dp[i][x], dp[i - 1][y]);
                else
                    dp[i][x] = min(dp[i][x], dp[i - 1][y] + val[i]);
            }
        }
        return min(dp[n][0], dp[n][1]);
    };
    int t1 = f(a);
    ff(i, 1, n) ff(j, i + 1, n) swap(maps[i][j], maps[j][i]);
    int t2 = f(b);
    cout << ((t1 + t2) >= INF / 2 ? -1 : t1 + t2) << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}