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
#define INF 0x3f3f3f3f
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
const int N = 2e5 + 5;
// int std::gcd(int a, int b) { return b ? std::gcd(b, a % b) : a; }
void Murasame()
{
    int n;
    cin >> n;
    vi a(n);
    ff(i, 0, n - 1)
    {
        cin >> a[i];
    }
    int temp = a[0];
    ff(i, 1, n - 1)
    {
        temp = std::gcd(temp, a[i]);
    }
    int ok = 0;
    for (int x : a)
    {
        if (x == temp)
            ok++;
    }
    if (ok > 0)
    {
        cout << (n - ok) << endl;
        return;
    }
    sort(all(a));
    a.erase(unique(all(a)), a.end());
    vi g = a;
    int dist[5000 + 1];
    for (int v = 1; v <= 5000; v++)
    {
        dist[v] = INF;
    }
    queue<int> q;
    for (int u : g)
    {
        if (dist[u] > 1)
        {
            dist[u] = 1;
            q.push(u);
        }
    }
    while (!q.empty())
    {
        int v = q.front();
        q.pop();
        int dv = dist[v];
        if (v == temp)
            break;
        for (int u : g)
        {
            int w = std::gcd(v, u);
            if (dist[w] > dv + 1)
            {
                dist[w] = dv + 1;
                q.push(w);
            }
        }
    }
    int k = dist[temp];
    int ans = n + k - 2;
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