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
int mod = 1e8 - 3;
const string AC = "Accepted";
bool cmp(pii a, pii b)
{
    if (a.fi == b.fi)
        return a.se < b.se;
    return a.fi < b.fi;
}
void Murasame()
{
    int n, ans = 0;
    cin >> n;
    vc<pii> a(n + 1), b(n + 1);
    ff(i, 1, n)
    {
        cin >> a[i].fi;
        a[i].se = i;
    }
    ff(i, 1, n)
    {
        cin >> b[i].fi;
        b[i].se = i;
    }
    sort(all1(a), cmp);
    sort(all1(b), cmp);
    vi rank(n + 1), tree(n + 1);
    ff(i, 1, n)
        rank[a[i].se] = b[i].se;
    auto add = [&](int x, int val)
    {
        while (x <= n)
        {
            tree[x] += val;
            tree[x] %= mod;
            x += x & -x;
        }
    };
    auto sum = [&](int x)
    {
        int res = 0;
        while (x)
        {
            res += tree[x];
            res %= mod;
            x -= x & -x;
        }
        return res;
    };
    ff(i, 1, n)
    {
        add(rank[i], 1);
        ans = (ans + i - sum(rank[i])) % mod;
    }
    cout << ans << endl;
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