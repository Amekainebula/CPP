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
    int n, m;
    cin >> n >> m;
    vi a(n), b(m);
    ff(i, 0, n - 1) cin >> a[i];
    ff(i, 0, m - 1) cin >> b[i];
    vi pre(n + 1, 0);
    ff(i, 1, n)
    {
        pre[i] = pre[i - 1];
        if (pre[i - 1] < m && a[i - 1] >= b[pre[i - 1]])
            pre[i]++;
    }
    if (pre[n] >= m)
    {
        cout << 0 << '\n';
        return;
    }
    vi s(n + 1, 0);
    ffg(i, n - 1, 0)
    {
        s[i] = s[i + 1];
        if (s[i + 1] < m && a[i] >= b[m - 1 - s[i + 1]])
            s[i]++;
    }
    int mink = inf;
    ff(j, 0, m - 1)
    {
        if (pre[n] < j)
            continue;
        int q = lower_bound(pre.begin(), pre.end(), j) - pre.begin();
        if (q > n)
            continue;
        int temp = m - 1 - j;
        if (s[q] >= temp)
            mink = min(mink, b[j]);
    }
    if (mink != inf)
    {
        cout << mink << endl;
    }
    else
    {
        cout << -1 << endl;
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