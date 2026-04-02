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
void solve()
{
    int n;
    pii man;
    cin >> n;
    cin >> man.fi >> man.se;
    string s;
    cin >> s;
    pii f = {0, 0};
    set<pii> st;
    st.insert(f);
    for (int i = 0; i < n; i++)
    {
        if (s[i] == 'N')
        {
            man.fi++;
            f.fi++;
        }
        else if (s[i] == 'S')
        {
            man.fi--;
            f.fi--;
        }
        else if (s[i] == 'W')
        {
            man.se++;
            f.se++;
        }
        else if (s[i] == 'E')
        {
            man.se--;
            f.se--;
        }
        st.insert(f);
        if (st.find(man) != st.end())
            cout << 1;
        else
            cout << 0;
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
        solve();
    }
    return 0;
}