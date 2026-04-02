#include <bits/stdc++.h>
// #define int long long
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
// #define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int N = 1e6 + 10;
// int n, m, k, x, y, z, ans, q, l, r, a[N], pos[N];

void Murasame()
{
    int n;
    cin >> n;
    fflush(stdout);
    vi ans(n + 1);
    vi g[n + 1];
    ff(i, 1, n - 1)
    {
        int u, v;
        cin >> u >> v;
        g[u].pb(v);
    }
    int rv = inf;
    int v1 = inf;
    bool ok = 0;
    cout << "? 1 1 1" << endl;
    fflush(stdout);
    int temp;
    cin >> temp;
    fflush(stdout);
    if (temp == 2 || temp == -2)
    {
        rv = temp / 2;
        v1 = temp / 2;
    }
    else if (temp == 0)
    {
        cout << "? 2 1" << endl;
        fflush(stdout);
        cout << "? 1 1 1" << endl;
        fflush(stdout);
        cin >> temp;
        fflush(stdout);
        rv = temp / 2;
        v1 = temp / 2;
    }
    else
    {
        rv = v1 = temp;
        ok = 1;
    }
    ans[1] = v1;
    if (ok)
    {
        for (int i = 2; i <= n; i++)
        {
            cout << "? " << 1 << " " << 1 << " " << i << endl;
            fflush(stdout);
            cin >> temp;
            fflush(stdout);
            ans[i] = temp - ans[1];
        }
    }
    else
    {
        int tt = rv + v1;
        for (int i = 2; i <= n; i++)
        {
            cout << "? " << 1 << " " << 1 << " " << i << endl;
            fflush(stdout);
            cin >> temp;
            fflush(stdout);
            int q = temp - tt;
            if (temp == rv && !ok)
            {
                cout << "? " << 2 << " " << i << endl;
                fflush(stdout);
                int temp2;
                cin >> temp2;
                if (temp2 == -rv)
                {
                    ok = 1;
                    rv = -rv;
                    ans[i] = rv;
                    tt = rv + v1;
                }
                else
                {
                    ans[i] = temp2 - rv - v1;
                }
            }
            else
            {
                ans[i] = q;
            }
        }
    }
    cout << "! ";
    ff(i, 1, n)cout << ans[i] << " ";
    cout << endl;
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