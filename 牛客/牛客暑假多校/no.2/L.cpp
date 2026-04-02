#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
// #define endl endl << flush
#define endl '\n'
using namespace std;
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 1e6 + 6;
int qsm(int a, int b)
{
    int res = 1;
    while (b)
    {
        if (b & 1)
            res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1), vis(n + 1, 0);
    ff(i, 1, n)
    {
        cin >> a[i];
    }
    vi ji, ou;
    auto dfs = [&](auto &&dfs, int u, int cnt) -> void
    {
        if (vis[u])
        {
            if (cnt & 1)
            {
                ji.pb(cnt);
            }
            else
            {
                ou.pb(cnt);
            }
            return;
        }
        else
        {
            vis[u] = 1;
            dfs(dfs, a[u], cnt + 1);
        }
    };
    ff(i, 1, n)
    {
        if (!vis[i])
        {
            dfs(dfs, i, 0);
        }
    }
    int cnt = 0;
    for (auto x : ou)
    {
        if (x > 2)
            cnt++;
    }
    if (ji.size() == 2)
    {
        cout << ji[0] * ji[1] % mod * qsm(2, cnt) % mod << endl;
    }
    else if (ji.size() == 0)
    {
        int ans = 0;
        for (auto x : ou)
        {
            ans = (ans + x * x % mod * qsm(4, mod - 2) % mod * qsm(2, cnt - (x != 2))) % mod;
        }
        cout << ans << endl;
    }
    else
    {
        cout << 0 << endl;
    }
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