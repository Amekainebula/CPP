#include <bits/stdc++.h>
// #define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
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
int fa[300005];
int finds(int x)
{
    return fa[x] == x ? x : fa[x] = finds(fa[x]);
}
void merge(int x, int y)
{
    int fx = finds(x);
    int fy = finds(y);
    if (fx != fy)
        fa[fx] = fy;
}

void solve()
{
    int n;
    cin >> n;
    vc<int> dis(n + 1);
    // set<int> ans;
    ff(i, 1, n) fa[i] = i;
    ff(i, 1, n)
    {
        cin >> dis[i];
        int t1 = i + dis[i], t2 = i - dis[i];
        if (t1 <= n)
            merge(i, t1);
        if (t2 >= 1)
            merge(i, t2);
    }
    map<int, int> mp;
    ff(i, 1, n)
        mp[finds(i)]++;
    cout << mp.size() - 1 << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}