#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define vi vector<int>
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
bool cmp(array<int, 2> a, array<int, 2> b)
{
    return a[0] == b[0] ? a[1] < b[1] : a[0] < b[0];
}
void Murasame()
{
    int n;
    cin >> n;
    vi tree(n + 1), rank(n + 1);
    vc<array<int, 2>> a(n + 1);
    ff(i, 1, n)
    {
        cin >> a[i][0];
        a[i][1] = i;
    }
    sort(a.begin() + 1, a.end(), cmp);
    auto add = [&](int x, int val)
    {
        while (x <= n)
        {
            tree[x] += val;
            x += (x & -x);
        }
    };
    auto sum = [&](int x)
    {
        int res = 0;
        while (x)
        {
            res += tree[x];
            x -= (x & -x);
        }
        return res;
    };
    int ans = 0;
    ff(i, 1, n)
    {
        rank[a[i][1]] = i;
    }
    ff(i, 1, n)
    {
        add(rank[i], 1);
        ans += i - sum(rank[i]);
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