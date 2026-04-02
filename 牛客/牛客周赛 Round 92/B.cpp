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
    vc<vc<char>> g(n + 1, vc<char>(m + 1));
    ff(i, 1, n) ff(j, 1, m) cin >> g[i][j];
    bool ok = 1;
    ff(i, 1, m) if (g[1][i] != '.') ok = 0;
    ff(i, 1, n) if (g[i][m] != '.') ok = 0;
    if (ok)
    {
        ff(i, 1, m - 1)
                cout
            << "D";
        ff(i, 1, n - 1)
                cout
            << "S";
    }
    else
    {
        ff(i, 1, n - 1)
                cout
            << "S";
        ff(i, 1, m - 1)
                cout
            << "D";
    }
    cout << endl;
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