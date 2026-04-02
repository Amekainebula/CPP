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
#define vvi vector<vi>
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
    int n, k;
    cin >> n >> k;
    if (k <= n - 1 || k >= n * n - (n - 2))
    {
        cout << "No" << endl;
        return;
    }
    cout << "Yes" << endl;
    vc<vc<int>> a(n + 1, vi(n + 1, 0));
    vi vis(n * n + 100, 0);
    int now = 0;
    a[1][1] = k;
    vis[k] = 1;
    ff(i, 2, n)
    {
        a[i][1] = ++now;
        vis[now] = 1;
    }
    ff(i, 2, n)
    {
        a[i][i] = n * n - i + 2;
        vis[n * n - i + 2] = 1;
    }
    ff(i, 1, n)
    {
        ff(j, 1, n)
        {
            if (a[i][j] == 0)
            {
                while (vis[now])
                    now++;
                a[i][j] = now;
                vis[now] = 1;
            }
        }
    }
    ff(i, 1, n) ff(j, 1, n) cout << a[i][j] << " \n"[j == n];
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