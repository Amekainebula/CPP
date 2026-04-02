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
int qpow(int a, int b)
{
    int ans = 1;
    a %= mod;
    while (b)
    {
        if (b & 1)
            ans = (ans * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return ans;
}
void Murasame()
{
    int n, m;
    cin >> n >> m;
    vc<pii> a(n + 1);
    ff(i, 1, n)
    {
        cin >> a[i].fi >> a[i].se;
    }
    auto cmp = [](pii a, pii b)
    { return a.se < b.se; };
    sort(all1(a), cmp);
    vc<pii> b;
    b.eb(-100, -100);
    ff(i, 1, n)
    {
        if (b.size() == 1 || b.back().se != a[i].se)
        {
            b.eb(a[i]);
        }
        else
        {
            b.back().fi += a[i].fi;
        }
    }
    n = b.size() - 1;
    int now = 0, ans = 0;
    bool ok = 0;
    ffg(i, n, 1)
    {
        if (i < n)
        {
            int temp = b[i + 1].se - b[i].se;
            ff(j, 1, temp)
            {
                now <<= 1;
                if (now > 1e18)
                {
                    ok = 1;
                    break;
                }
            }
            if (ok)
                break;
        }
        if (now >= b[i].fi)
        {
            now -= b[i].fi;
            b[i].fi = 0;
        }
        else
        {
            b[i].fi -= now;
            now = 0;
        }
        if (b[i].fi)
        {
            int has = b[i].fi / m;
            ans = (ans + has % mod * qpow(2, b[i].se) % mod) % mod;
            if (b[i].fi % m)
            {
                ans = (ans + qpow(2, b[i].se) % mod) % mod;
                now = m - b[i].fi % m;
            }
        }
    }
    cout << ans << endl;
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